import unittest

from Cluster.topology import Link


class TestLinkFlowCount(unittest.TestCase):
    def _make_link(self, name: str = "link-A:link-B") -> Link:
        return Link(name=name, src_port=None, dst_port=None, speed_gbps=100.0)

    def test_initial_flow_count_is_zero(self):
        """A freshly constructed Link starts with flow_count == 0."""
        link = self._make_link()
        self.assertEqual(link.flow_count, 0)

    def test_incFlow_advances_by_one(self):
        """incFlow increments by exactly 1 each call."""
        link = self._make_link()
        link.incFlow()
        self.assertEqual(link.flow_count, 1)
        for _ in range(4):
            link.incFlow()
        self.assertEqual(link.flow_count, 5)

    def test_decFlow_returns_to_zero_after_inc(self):
        """incFlow then decFlow leaves flow_count at 0."""
        link = self._make_link()
        link.incFlow()
        link.decFlow()
        self.assertEqual(link.flow_count, 0)

    def test_decFlow_on_fresh_link_raises(self):
        """decFlow on a link at flow_count == 0 raises ValueError naming the link."""
        link = self._make_link(name="x0-y0-p1:x1-y0-p0")
        with self.assertRaises(ValueError) as cm:
            link.decFlow()
        self.assertIn("x0-y0-p1:x1-y0-p0", str(cm.exception))

    def test_decFlow_after_balanced_pair_raises(self):
        """Once flow_count returns to 0 after an inc/dec pair, decFlow raises again."""
        link = self._make_link()
        link.incFlow()
        link.decFlow()
        self.assertEqual(link.flow_count, 0)
        with self.assertRaises(ValueError):
            link.decFlow()


if __name__ == "__main__":
    unittest.main()
