'Regression tests for the native Ristretto255 blind OPRF.'

from __future__ import annotations

import sys
import tempfile
import unittest
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dbtfl.communication import JsonHttpClient, RemoteServiceError, ThreadedJsonServer, WireMessage
from dbtfl.communication.endpoints import KeyServerPath
from dbtfl.oprf import (
    KeyServerOprfService,
    OprfClient,
    OprfKeyStore,
    OprfKeyStoreError,
    native_backend_available,
    validate_protected_label,
)
from dbtfl.oprf.service import OPRF_EVALUATE_REQUEST
from dbtfl.entities import ClientConfig, ClientEntity, ClientLabelStoreError


class OprfMigrationTest(unittest.TestCase):
    'Verify legacy MODP material cannot silently enter the new suite.'

    def test_legacy_key_is_rejected_before_any_crypto_operation(self) -> None:
        'Require an explicit new Ristretto255 key path for migration.'
        with tempfile.TemporaryDirectory() as temporary:
            key_path = Path(temporary) / "legacy-key.json"
            key_path.write_text(
                json.dumps({"version": 1, "group": "rfc3526-modp-group14-q-subgroup", "k": "5"}),
                encoding="utf-8",
            )
            with self.assertRaises(OprfKeyStoreError):
                OprfKeyStore(key_path).load_or_create()

    def test_legacy_label_store_is_rejected_before_ks_contact(self) -> None:
        'Require a fresh cache rather than pairing old labels with a new key.'
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store_path = root / "legacy-labels.json"
            store_path.write_text(
                json.dumps({"schema_version": "1.0", "client_id": "client-0", "entries": []}),
                encoding="utf-8",
            )
            with self.assertRaises(ClientLabelStoreError):
                ClientEntity(ClientConfig(
                    client_id="client-0",
                    ks_base_url="http://127.0.0.1:1",
                    label_store_path=store_path,
                ))


@unittest.skipUnless(native_backend_available(), "requires oblivious with native libsodium Ristretto255")
class OprfIntegrationTest(unittest.TestCase):
    'Exercise blind evaluation through a real loopback KS listener.'

    def setUp(self) -> None:
        'Create an ephemeral suite-bound key and local KS listener.'
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary_directory.cleanup)
        self._key_store = OprfKeyStore(Path(self._temporary_directory.name) / "ks-ristretto255-key.json")
        self._service = KeyServerOprfService(self._key_store.load_or_create())
        self._server = ThreadedJsonServer("127.0.0.1", 0, self._service.router)
        self._server.start()
        self.addCleanup(self._server.close)
        self._client = OprfClient(JsonHttpClient(self._server.base_url), batch_size=3)

    def test_labels_are_deterministic_and_keep_native_index_contract(self) -> None:
        'Verify independent blinds lead to one stable 512-hex label.'
        record = "cross-platform federation record"
        first = self._client.evaluate([record])[0]
        second = self._client.evaluate([record])[0]
        other = self._client.evaluate(["different private record"])[0]

        self.assertEqual(first, second)
        self.assertNotEqual(first, other)
        self.assertEqual(validate_protected_label(first), first)

    def test_batches_preserve_input_order(self) -> None:
        'Verify wire batching neither reorders nor changes OPRF outputs.'
        records = [f"private record {index}" for index in range(8)]
        labels = self._client.evaluate(records)
        self.assertEqual(labels, [self._client.evaluate([record])[0] for record in records])

    def test_ks_rejects_malformed_wire_points(self) -> None:
        'Verify the KS rejects malformed point encodings before evaluation.'
        request = WireMessage.create(OPRF_EVALUATE_REQUEST, {"blinded_elements": ["not-base64"]})
        with self.assertRaises(RemoteServiceError) as captured:
            JsonHttpClient(self._server.base_url).send(KeyServerPath.EVALUATE_OPRF.value, request)
        self.assertEqual(captured.exception.status_code, 400)
        self.assertEqual(captured.exception.code, "invalid_blinded_element")

    def test_existing_key_is_reused_without_rotation(self) -> None:
        'Verify the Ristretto255 scalar remains stable across KS restarts.'
        self.assertEqual(self._key_store.load_or_create(), self._key_store.load_or_create())


if __name__ == "__main__":
    unittest.main()
