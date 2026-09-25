'Cross-platform integration tests for KS and client role entities.\nKS'

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dbtfl.communication import TransportError
from dbtfl.entities import ClientConfig, ClientEntity, KeyServerConfig, KeyServerEntity


class RoleEntityIntegrationTest(unittest.TestCase):
    'Exercise private client persistence against a real loopback KS.'

    def setUp(self) -> None:
        'Create a temporary Ristretto255 KS identity.'
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary_directory.cleanup)
        self._root = Path(self._temporary_directory.name)
        key_path = self._root / "ks-key.json"
        self._ks = KeyServerEntity(
            KeyServerConfig(
                key_path=key_path,
                host="127.0.0.1",
                port=0,
            )
        )
        self._ks.start()
        self.addCleanup(self._ks.close)

    def test_key_server_config_defaults_to_port_18081(self) -> None:
        'Verify the deployment configuration preserves the documented KS port.'
        config = KeyServerConfig(key_path=self._root / "another-ks-key.json")

        self.assertEqual(config.port, 18081)

    def test_client_persists_recoverable_record_label_correspondence(self) -> None:
        'Verify one client reuses a local mapping without another KS request.'
        store_path = self._root / "client-labels.json"
        client = ClientEntity(
            ClientConfig(
                client_id="client-1",
                ks_base_url=self._ks.base_url,
                label_store_path=store_path,
            )
        )
        labels = client.generate_protected_labels(["alpha", b"beta", "alpha"])

        self.assertEqual(labels[0], labels[2])
        self.assertEqual(client.record_count, 2)
        self.assertEqual(client.protected_label_for("alpha"), labels[0])
        self.assertEqual(client.protected_label_for(b"beta"), labels[1])
        store_text = store_path.read_text(encoding="utf-8")
        self.assertNotIn("alpha", store_text)
        self.assertNotIn("beta", store_text)

        self._ks.close()
        recovered_client = ClientEntity(
            ClientConfig(
                client_id="client-1",
                ks_base_url="http://127.0.0.1:1",
                label_store_path=store_path,
            )
        )
        self.assertEqual(recovered_client.generate_protected_labels(["alpha", b"beta"]), labels[:2])

    def test_client_does_not_write_partial_mapping_when_ks_is_unreachable(self) -> None:
        'Verify failed OPRF transport leaves an absent store absent.'
        store_path = self._root / "unwritten-labels.json"
        client = ClientEntity(
            ClientConfig(
                client_id="client-2",
                ks_base_url="http://127.0.0.1:1",
                label_store_path=store_path,
                timeout_seconds=0.1,
            )
        )

        with self.assertRaises(TransportError):
            client.generate_protected_labels(["unreachable record"])

        self.assertFalse(store_path.exists())
        self.assertEqual(client.record_count, 0)



if __name__ == "__main__":
    unittest.main()
