import simpy
import unittest

from common.flags import *
from common.job import TopoType
from Cluster.cluster import Cluster
from Cluster.model_builder import build


class TestModel(unittest.TestCase):

    def setUp(self):
        self.env = simpy.Environment()

    def test_t2d(self):
        """
        Test properties of a 4x4 2D Torus model.
        """
        model = build(topo=TopoType.T2D, name="test", dimension=(4, 4), xpu_per_node=1)
        cluster = Cluster(self.env, spec=model)
        self.assertIsNotNone(model)
        self.assertIsNotNone(cluster)
        self.assertEqual(model["name"], "test")
        self.assertEqual(model["topology"], "T2D")
        self.assertEqual(model["dimx"], 4)
        self.assertEqual(model["dimy"], 4)
        self.assertEqual(model["dimz"], 0)
        self.assertEqual(model["total_nodes"], 4 * 4)
        self.assertEqual(len(model["nodes"]), 4 * 4)
        for node in model["nodes"]:
            coord = node["coordinates"]
            self.assertEqual(node["name"], f"x{coord[0]}-y{coord[1]}")
        self.assertEqual(cluster.name, "test")
        self.assertEqual(cluster.topo, TopoType.T2D)
        self.assertEqual(cluster.numNodes(), 4 * 4)
        for node in cluster.allNodes().values():
            self.assertEqual(node.numXPU(), 1)
            self.assertEqual(node.numIdleXPU(), 1)
            self.assertEqual(cluster.getIdleXPU(node.name), 1)

    def test_t2d_bad(self):
        """
        Test building a bad 4x4 2D Torus model.
        """
        self.assertRaises(ValueError, build, TopoType.T2D, "test", (4, 4, 4), 1)
        self.assertRaises(ValueError, build, TopoType.T2D, "test", (0, 0), 1)
        self.assertRaises(ValueError, build, TopoType.T2D, "test", (4, -1), 1)
        self.assertRaises(ValueError, build, TopoType.T2D, "test", (4, 4.1), 1)
        self.assertRaises(ValueError, build, TopoType.T2D, "test", (4, 4), 2)

    def test_t3d(self):
        """
        Test properties of a 4x4x4 3D Torus model.
        """
        model = build(
            topo=TopoType.T3D_NT, name="test", dimension=(4, 4, 4), xpu_per_node=1
        )
        cluster = Cluster(self.env, spec=model)
        self.assertIsNotNone(model)
        self.assertIsNotNone(cluster)
        self.assertEqual(model["name"], "test")
        self.assertEqual(model["topology"], "T3D_NT")
        self.assertEqual(model["dimx"], 4)
        self.assertEqual(model["dimy"], 4)
        self.assertEqual(model["dimz"], 4)
        self.assertEqual(model["total_nodes"], 4 * 4 * 4)
        self.assertEqual(len(model["nodes"]), 4 * 4 * 4)
        for node in model["nodes"]:
            coord = node["coordinates"]
            self.assertEqual(node["name"], f"x{coord[0]}-y{coord[1]}-z{coord[2]}")
        self.assertEqual(cluster.name, "test")
        self.assertEqual(cluster.topo, TopoType.T3D_NT)
        self.assertEqual(cluster.numNodes(), 4 * 4 * 4)
        for node in cluster.allNodes().values():
            self.assertEqual(node.numXPU(), 1)
            self.assertEqual(node.numIdleXPU(), 1)
            self.assertEqual(cluster.getIdleXPU(node.name), 1)

    def test_t3d_bad(self):
        """
        Test building a bad 4x4x4 3D Torus model.
        """
        self.assertRaises(ValueError, build, TopoType.T3D_NT, "test", (4, 4), 1)
        self.assertRaises(ValueError, build, TopoType.T3D_NT, "test", (4, -1, 0), 1)
        self.assertRaises(ValueError, build, TopoType.T3D_NT, "test", (4, 1, 0.5), 1)
        self.assertRaises(ValueError, build, TopoType.T3D_NT, "test", (4, 4, 4), 2)

    def test_clos_1tier(self):
        """
        Test properties of a Clos model.
        """
        model = build(topo=TopoType.CLOS, name="test", dimension=(8, 1), xpu_per_node=8)
        cluster = Cluster(self.env, spec=model)
        self.assertIsNotNone(model)
        self.assertIsNotNone(cluster)
        self.assertEqual(model["name"], "test")
        self.assertEqual(model["topology"], "CLOS")
        self.assertEqual(model["dimx"], 8)
        self.assertEqual(model["dimy"], 1)
        self.assertEqual(model["dimz"], 0)
        self.assertEqual(model["total_nodes"], 8 * 1)
        self.assertEqual(len(model["nodes"]), 8 * 1)
        for node in model["nodes"]:
            coord = node["coordinates"]
            self.assertEqual(node["name"], f"t{coord[1]}-n{coord[0]}")
        self.assertEqual(cluster.name, "test")
        self.assertEqual(cluster.topo, TopoType.CLOS)
        self.assertEqual(cluster.numNodes(), 8 * 1)
        for node in cluster.allNodes().values():
            self.assertEqual(node.numXPU(), 8)
            self.assertEqual(node.numIdleXPU(), 8)
            self.assertEqual(cluster.getIdleXPU(node.name), 8)
