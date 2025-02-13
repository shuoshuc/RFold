import simpy
import unittest

from common.job import TopoType
from Cluster.cluster import Cluster
from Cluster.model_builder import build, connect_links


class TestLinkConstruction(unittest.TestCase):

    def test_connect_links_mesh2d(self):
        """
        Verify that links are constructed correctly for 2D mesh.
        """
        # Connect node x0-y0 in a 2D mesh.
        links = connect_links(
            topo=TopoType.MESH2D,
            x=0,
            y=0,
            z=None,
            dimx=4,
            dimy=4,
            dimz=None,
            speed_gbps=100,
        )
        self.assertEqual(len(links), 6)
        # No link on the x- direction.
        self.assertIsNone(links[0])
        # Has link on the x+ direction.
        self.assertIsNotNone(links[1])
        src, dst = "x0-y0-p1", "x1-y0-p0"
        self.assertEqual(links[1]["name"], f"{src}:{dst}")
        self.assertEqual(links[1]["src"], src)
        self.assertEqual(links[1]["dst"], dst)
        self.assertEqual(links[1]["speed_gbps"], 100)
        # No link on the y- direction.
        self.assertIsNone(links[2])
        # Has link on the y+ direction.
        self.assertIsNotNone(links[3])
        src, dst = "x0-y0-p3", "x0-y1-p2"
        self.assertEqual(links[3]["name"], f"{src}:{dst}")
        self.assertEqual(links[3]["src"], src)
        self.assertEqual(links[3]["dst"], dst)
        self.assertEqual(links[3]["speed_gbps"], 100)
        # No link on the z-/z+ direction.
        self.assertIsNone(links[4])
        self.assertIsNone(links[5])

    def test_connect_links_t2d(self):
        """
        Verify that links are constructed correctly for 2D torus.
        """
        # Connect node x0-y0 in a 2D torus.
        links = connect_links(
            topo=TopoType.T2D,
            x=0,
            y=0,
            z=None,
            dimx=4,
            dimy=4,
            dimz=None,
            speed_gbps=100,
        )
        self.assertEqual(len(links), 6)
        src_dst_pairs = [
            ("x0-y0-p0", "x3-y0-p1"),  # x-
            ("x0-y0-p1", "x1-y0-p0"),  # x+
            ("x0-y0-p2", "x0-y3-p3"),  # y-
            ("x0-y0-p3", "x0-y1-p2"),  # y+
        ]
        # Has links in x-, x+, y-, y+ directions.
        for i in range(4):
            self.assertIsNotNone(links[i])
            src, dst = src_dst_pairs[i]
            self.assertEqual(links[i]["name"], f"{src}:{dst}")
            self.assertEqual(links[i]["src"], src)
            self.assertEqual(links[i]["dst"], dst)
            self.assertEqual(links[i]["speed_gbps"], 100)
        # No link on the z-/z+ direction.
        self.assertIsNone(links[4])
        self.assertIsNone(links[5])

    def test_connect_links_mesh3d(self):
        """
        Verify that links are constructed correctly for 3D mesh.
        """
        # Connect node x0-y0-z0 in a 3D mesh.
        links = connect_links(
            topo=TopoType.MESH3D,
            x=0,
            y=0,
            z=0,
            dimx=4,
            dimy=4,
            dimz=4,
            speed_gbps=100,
        )
        self.assertEqual(len(links), 6)
        # No link on the x- direction.
        self.assertIsNone(links[0])
        # Has link on the x+ direction.
        self.assertIsNotNone(links[1])
        src, dst = "x0-y0-z0-p1", "x1-y0-z0-p0"
        self.assertEqual(links[1]["name"], f"{src}:{dst}")
        self.assertEqual(links[1]["src"], src)
        self.assertEqual(links[1]["dst"], dst)
        self.assertEqual(links[1]["speed_gbps"], 100)
        # No link on the y- direction.
        self.assertIsNone(links[2])
        # Has link on the y+ direction.
        self.assertIsNotNone(links[3])
        src, dst = "x0-y0-z0-p3", "x0-y1-z0-p2"
        self.assertEqual(links[3]["name"], f"{src}:{dst}")
        self.assertEqual(links[3]["src"], src)
        self.assertEqual(links[3]["dst"], dst)
        self.assertEqual(links[3]["speed_gbps"], 100)
        # No link on the z- direction.
        self.assertIsNone(links[4])
        # Has link on the z+ direction.
        self.assertIsNotNone(links[5])
        src, dst = "x0-y0-z0-p5", "x0-y0-z1-p4"
        self.assertEqual(links[5]["name"], f"{src}:{dst}")
        self.assertEqual(links[5]["src"], src)
        self.assertEqual(links[5]["dst"], dst)
        self.assertEqual(links[5]["speed_gbps"], 100)

    def test_connect_links_t3d(self):
        """
        Verify that links are constructed correctly for 3D torus.
        """
        # Unable to connect links in twisted 3D torus.
        self.assertRaises(
            NotImplementedError, connect_links, TopoType.T3D_T, 0, 0, 0, 4, 4, 4, 100
        )

        # Connect node x0-y0-z0 in a non-twisted 3D torus.
        links = connect_links(
            topo=TopoType.T3D_NT,
            x=0,
            y=0,
            z=0,
            dimx=4,
            dimy=4,
            dimz=4,
            speed_gbps=100,
        )
        self.assertEqual(len(links), 6)
        src_dst_pairs = [
            ("x0-y0-z0-p0", "x3-y0-z0-p1"),  # x-
            ("x0-y0-z0-p1", "x1-y0-z0-p0"),  # x+
            ("x0-y0-z0-p2", "x0-y3-z0-p3"),  # y-
            ("x0-y0-z0-p3", "x0-y1-z0-p2"),  # y+
            ("x0-y0-z0-p4", "x0-y0-z3-p5"),  # z-
            ("x0-y0-z0-p5", "x0-y0-z1-p4"),  # z+
        ]
        # Has links in all directions.
        for i in range(6):
            self.assertIsNotNone(links[i])
            src, dst = src_dst_pairs[i]
            self.assertEqual(links[i]["name"], f"{src}:{dst}")
            self.assertEqual(links[i]["src"], src)
            self.assertEqual(links[i]["dst"], dst)
            self.assertEqual(links[i]["speed_gbps"], 100)


class TestModelSpec(unittest.TestCase):

    def test_mesh2d_spec(self):
        """
        Test properties of a 4x4 2D mesh model.
        """
        model = build(
            topo=TopoType.MESH2D,
            name="test",
            dimension=(4, 4),
            xpu_per_node=1,
            port_per_node=4,
            port_speed_gbps=100,
        )
        self.assertIsNotNone(model)
        self.assertEqual(model["name"], "test")
        self.assertEqual(model["topology"], "MESH2D")
        self.assertEqual(model["dimx"], 4)
        self.assertEqual(model["dimy"], 4)
        self.assertEqual(model["dimz"], 0)
        self.assertEqual(model["total_nodes"], 4 * 4)
        self.assertEqual(len(model["nodes"]), 4 * 4)
        # x dimension: 3x2 links per row, 4 rows. Same for y dimension.
        self.assertEqual(len(model["links"]), 4 * 3 * 2 + 4 * 3 * 2)
        for node in model["nodes"]:
            coord = node["coordinates"]
            self.assertEqual(node["name"], f"x{coord[0]}-y{coord[1]}")
            self.assertEqual(node["num_xpu"], 1)
            self.assertEqual(len(node["ports"]), 4)
            for port in node["ports"]:
                self.assertLess(port["index"], 4)
                self.assertEqual(port["name"], f"{node['name']}-p{port['index']}")
                self.assertEqual(port["speed_gbps"], 100)
        X_LINK_1 = {
            "name": "x1-y0-p1:x2-y0-p0",
            "src": "x1-y0-p1",
            "dst": "x2-y0-p0",
            "speed_gbps": 100,
        }
        X_LINK_2 = {
            "name": "x2-y0-p0:x1-y0-p1",
            "src": "x2-y0-p0",
            "dst": "x1-y0-p1",
            "speed_gbps": 100,
        }
        Y_LINK_1 = {
            "name": "x2-y1-p3:x2-y2-p2",
            "src": "x2-y1-p3",
            "dst": "x2-y2-p2",
            "speed_gbps": 100,
        }
        Y_LINK_2 = {
            "name": "x2-y2-p2:x2-y1-p3",
            "src": "x2-y2-p2",
            "dst": "x2-y1-p3",
            "speed_gbps": 100,
        }
        # There should be a pair of links in each direction between 2 nodes.
        self.assertIn(X_LINK_1, model["links"])
        self.assertIn(X_LINK_2, model["links"])
        self.assertIn(Y_LINK_1, model["links"])
        self.assertIn(Y_LINK_2, model["links"])

        # Verify that all x-dimension wrap-around links are *NOT* present.
        for y in range(4):
            x_wrap_link = {
                "name": f"x3-y{y}-p1:x0-y{y}-p0",
                "src": f"x3-y{y}-p1",
                "dst": f"x0-y{y}-p0",
                "speed_gbps": 100,
            }
            self.assertNotIn(x_wrap_link, model["links"])
        # Verify that all y-dimension wrap-around links are *NOT* present.
        for x in range(4):
            y_wrap_link = {
                "name": f"x{x}-y0-p2:x{x}-y3-p3",
                "src": f"x{x}-y0-p2",
                "dst": f"x{x}-y3-p3",
                "speed_gbps": 100,
            }
            self.assertNotIn(y_wrap_link, model["links"])

        for link in model["links"]:
            self.assertEqual(link["speed_gbps"], 100)
            self.assertEqual(link["name"], f"{link['src']}:{link['dst']}")

    def test_t2d_spec(self):
        """
        Test properties of a 4x4 2D Torus model.
        """
        model = build(
            topo=TopoType.T2D,
            name="test",
            dimension=(4, 4),
            xpu_per_node=1,
            port_per_node=4,
            port_speed_gbps=100,
        )
        self.assertIsNotNone(model)
        self.assertEqual(model["name"], "test")
        self.assertEqual(model["topology"], "T2D")
        self.assertEqual(model["dimx"], 4)
        self.assertEqual(model["dimy"], 4)
        self.assertEqual(model["dimz"], 0)
        self.assertEqual(model["total_nodes"], 4 * 4)
        self.assertEqual(len(model["nodes"]), 4 * 4)
        # x dimension: 4x2 links per row, 4 rows. Same for y dimension.
        self.assertEqual(len(model["links"]), 4 * 4 * 2 + 4 * 4 * 2)
        for node in model["nodes"]:
            coord = node["coordinates"]
            self.assertEqual(node["name"], f"x{coord[0]}-y{coord[1]}")
            self.assertEqual(node["num_xpu"], 1)
            self.assertEqual(len(node["ports"]), 4)
            for port in node["ports"]:
                self.assertLess(port["index"], 4)
                self.assertEqual(port["name"], f"{node['name']}-p{port['index']}")
                self.assertEqual(port["speed_gbps"], 100)
        X_LINK_1 = {
            "name": "x1-y0-p1:x2-y0-p0",
            "src": "x1-y0-p1",
            "dst": "x2-y0-p0",
            "speed_gbps": 100,
        }
        X_LINK_2 = {
            "name": "x2-y0-p0:x1-y0-p1",
            "src": "x2-y0-p0",
            "dst": "x1-y0-p1",
            "speed_gbps": 100,
        }
        Y_LINK_1 = {
            "name": "x2-y1-p3:x2-y2-p2",
            "src": "x2-y1-p3",
            "dst": "x2-y2-p2",
            "speed_gbps": 100,
        }
        Y_LINK_2 = {
            "name": "x2-y2-p2:x2-y1-p3",
            "src": "x2-y2-p2",
            "dst": "x2-y1-p3",
            "speed_gbps": 100,
        }
        # There should be a pair of links in each direction between 2 nodes.
        self.assertIn(X_LINK_1, model["links"])
        self.assertIn(X_LINK_2, model["links"])
        self.assertIn(Y_LINK_1, model["links"])
        self.assertIn(Y_LINK_2, model["links"])

        # Verify that all x-dimension wrap-around links are present.
        for y in range(4):
            x_wrap_link = {
                "name": f"x3-y{y}-p1:x0-y{y}-p0",
                "src": f"x3-y{y}-p1",
                "dst": f"x0-y{y}-p0",
                "speed_gbps": 100,
            }
            self.assertIn(x_wrap_link, model["links"])
        # Verify that all y-dimension wrap-around links are present.
        for x in range(4):
            y_wrap_link = {
                "name": f"x{x}-y0-p2:x{x}-y3-p3",
                "src": f"x{x}-y0-p2",
                "dst": f"x{x}-y3-p3",
                "speed_gbps": 100,
            }
            self.assertIn(y_wrap_link, model["links"])

        for link in model["links"]:
            self.assertEqual(link["speed_gbps"], 100)
            self.assertEqual(link["name"], f"{link['src']}:{link['dst']}")

    def test_t2d_bad(self):
        """
        Test building a bad 4x4 2D Torus model. Omit tests for bad 2D mesh model because
        it works similarly.
        """
        GOOD_DIM = (4, 4)
        GOOD_XPU = 1
        GOOD_PORT = 4
        GOOD_SPEED = 100

        # Dimension must match topology type.
        self.assertRaises(
            ValueError,
            build,
            TopoType.T2D,
            "test",
            (4, 4, 4),
            GOOD_XPU,
            GOOD_PORT,
            GOOD_SPEED,
        )

        # Dimension should not contain non-positive values or fractions.
        self.assertRaises(
            ValueError,
            build,
            TopoType.T2D,
            "test",
            (0, 0),
            GOOD_XPU,
            GOOD_PORT,
            GOOD_SPEED,
        )
        self.assertRaises(
            ValueError,
            build,
            TopoType.T2D,
            "test",
            (4, -1),
            GOOD_XPU,
            GOOD_PORT,
            GOOD_SPEED,
        )
        self.assertRaises(
            ValueError,
            build,
            TopoType.T2D,
            "test",
            (4, 4.1),
            GOOD_XPU,
            GOOD_PORT,
            GOOD_SPEED,
        )

        # xpu_per_node should not be anything other than 1.
        self.assertRaises(
            ValueError, build, TopoType.T2D, "test", GOOD_DIM, 0, GOOD_PORT, GOOD_SPEED
        )
        self.assertRaises(
            ValueError, build, TopoType.T2D, "test", GOOD_DIM, 2, GOOD_PORT, GOOD_SPEED
        )

        # port_per_node should not be anything other than 4.
        self.assertRaises(
            ValueError, build, TopoType.T2D, "test", GOOD_DIM, GOOD_XPU, 3, GOOD_SPEED
        )
        self.assertRaises(
            ValueError, build, TopoType.T2D, "test", GOOD_DIM, GOOD_XPU, 0, GOOD_SPEED
        )

        # port_speed_gbps must be positive.
        self.assertRaises(
            ValueError, build, TopoType.T2D, "test", GOOD_DIM, GOOD_XPU, GOOD_PORT, -1
        )

    def test_mesh3d_spec(self):
        """
        Test properties of a 4x4x4 3D mesh model.
        """
        model = build(
            topo=TopoType.MESH3D,
            name="test",
            dimension=(4, 4, 4),
            xpu_per_node=1,
            port_per_node=6,
            port_speed_gbps=100,
        )
        self.assertIsNotNone(model)
        self.assertEqual(model["name"], "test")
        self.assertEqual(model["topology"], "MESH3D")
        self.assertEqual(model["dimx"], 4)
        self.assertEqual(model["dimy"], 4)
        self.assertEqual(model["dimz"], 4)
        self.assertEqual(model["total_nodes"], 4 * 4 * 4)
        self.assertEqual(len(model["nodes"]), 4 * 4 * 4)
        # x dimension: 3x2 links per row, 4x4 rows. Same for y, z dimensions.
        self.assertEqual(len(model["links"]), 4 * 4 * 3 * 2 * 3)
        for node in model["nodes"]:
            coord = node["coordinates"]
            self.assertEqual(node["name"], f"x{coord[0]}-y{coord[1]}-z{coord[2]}")
            self.assertEqual(node["num_xpu"], 1)
            self.assertEqual(len(node["ports"]), 6)
            for port in node["ports"]:
                self.assertLess(port["index"], 6)
                self.assertEqual(port["name"], f"{node['name']}-p{port['index']}")
                self.assertEqual(port["speed_gbps"], 100)
        X_LINK_1 = {
            "name": "x0-y0-z0-p1:x1-y0-z0-p0",
            "src": "x0-y0-z0-p1",
            "dst": "x1-y0-z0-p0",
            "speed_gbps": 100,
        }
        X_LINK_2 = {
            "name": "x1-y0-z0-p0:x0-y0-z0-p1",
            "src": "x1-y0-z0-p0",
            "dst": "x0-y0-z0-p1",
            "speed_gbps": 100,
        }
        Y_LINK_1 = {
            "name": "x0-y0-z0-p3:x0-y1-z0-p2",
            "src": "x0-y0-z0-p3",
            "dst": "x0-y1-z0-p2",
            "speed_gbps": 100,
        }
        Y_LINK_2 = {
            "name": "x0-y1-z0-p2:x0-y0-z0-p3",
            "src": "x0-y1-z0-p2",
            "dst": "x0-y0-z0-p3",
            "speed_gbps": 100,
        }
        Z_LINK_1 = {
            "name": "x0-y0-z0-p5:x0-y0-z1-p4",
            "src": "x0-y0-z0-p5",
            "dst": "x0-y0-z1-p4",
            "speed_gbps": 100,
        }
        Z_LINK_2 = {
            "name": "x0-y0-z1-p4:x0-y0-z0-p5",
            "src": "x0-y0-z1-p4",
            "dst": "x0-y0-z0-p5",
            "speed_gbps": 100,
        }
        # There should be a pair of links in each direction between 2 nodes.
        self.assertIn(X_LINK_1, model["links"])
        self.assertIn(X_LINK_2, model["links"])
        self.assertIn(Y_LINK_1, model["links"])
        self.assertIn(Y_LINK_2, model["links"])
        self.assertIn(Z_LINK_1, model["links"])
        self.assertIn(Z_LINK_2, model["links"])

        # Verify that all x-dimension wrap-around links are *NOT* present.
        for y in range(4):
            for z in range(4):
                x_wrap_link = {
                    "name": f"x3-y{y}-z{z}-p1:x0-y{y}-z{z}-p0",
                    "src": f"x3-y{y}-z{z}-p1",
                    "dst": f"x0-y{y}-z{z}-p0",
                    "speed_gbps": 100,
                }
                self.assertNotIn(x_wrap_link, model["links"])
        # Verify that all y-dimension wrap-around links are *NOT* present.
        for x in range(4):
            for z in range(4):
                y_wrap_link = {
                    "name": f"x{x}-y0-z{z}-p2:x{x}-y3-z{z}-p3",
                    "src": f"x{x}-y0-z{z}-p2",
                    "dst": f"x{x}-y3-z{z}-p3",
                    "speed_gbps": 100,
                }
                self.assertNotIn(y_wrap_link, model["links"])
        # Verify that all z-dimension wrap-around links are *NOT* present.
        for x in range(4):
            for y in range(4):
                z_wrap_link = {
                    "name": f"x{x}-y{y}-z0-p4:x{x}-y{y}-z3-p5",
                    "src": f"x{x}-y{y}-z0-p4",
                    "dst": f"x{x}-y{y}-z3-p5",
                    "speed_gbps": 100,
                }
                self.assertNotIn(z_wrap_link, model["links"])

        for link in model["links"]:
            self.assertEqual(link["speed_gbps"], 100)
            self.assertEqual(link["name"], f"{link['src']}:{link['dst']}")

    def test_t3d_spec(self):
        """
        Test properties of a 4x4x4 3D torus model.
        """
        model = build(
            topo=TopoType.T3D_NT,
            name="test",
            dimension=(4, 4, 4),
            xpu_per_node=1,
            port_per_node=6,
            port_speed_gbps=100,
        )
        self.assertIsNotNone(model)
        self.assertEqual(model["name"], "test")
        self.assertEqual(model["topology"], "T3D_NT")
        self.assertEqual(model["dimx"], 4)
        self.assertEqual(model["dimy"], 4)
        self.assertEqual(model["dimz"], 4)
        self.assertEqual(model["total_nodes"], 4 * 4 * 4)
        self.assertEqual(len(model["nodes"]), 4 * 4 * 4)
        # x dimension: 4x2 links per row, 4x4 rows. Same for y, z dimensions.
        self.assertEqual(len(model["links"]), 4 * 4 * 4 * 2 * 3)
        for node in model["nodes"]:
            coord = node["coordinates"]
            self.assertEqual(node["name"], f"x{coord[0]}-y{coord[1]}-z{coord[2]}")
            self.assertEqual(node["num_xpu"], 1)
            self.assertEqual(len(node["ports"]), 6)
            for port in node["ports"]:
                self.assertLess(port["index"], 6)
                self.assertEqual(port["name"], f"{node['name']}-p{port['index']}")
                self.assertEqual(port["speed_gbps"], 100)
        X_LINK_1 = {
            "name": "x0-y0-z0-p1:x1-y0-z0-p0",
            "src": "x0-y0-z0-p1",
            "dst": "x1-y0-z0-p0",
            "speed_gbps": 100,
        }
        X_LINK_2 = {
            "name": "x1-y0-z0-p0:x0-y0-z0-p1",
            "src": "x1-y0-z0-p0",
            "dst": "x0-y0-z0-p1",
            "speed_gbps": 100,
        }
        Y_LINK_1 = {
            "name": "x0-y0-z0-p3:x0-y1-z0-p2",
            "src": "x0-y0-z0-p3",
            "dst": "x0-y1-z0-p2",
            "speed_gbps": 100,
        }
        Y_LINK_2 = {
            "name": "x0-y1-z0-p2:x0-y0-z0-p3",
            "src": "x0-y1-z0-p2",
            "dst": "x0-y0-z0-p3",
            "speed_gbps": 100,
        }
        Z_LINK_1 = {
            "name": "x0-y0-z0-p5:x0-y0-z1-p4",
            "src": "x0-y0-z0-p5",
            "dst": "x0-y0-z1-p4",
            "speed_gbps": 100,
        }
        Z_LINK_2 = {
            "name": "x0-y0-z1-p4:x0-y0-z0-p5",
            "src": "x0-y0-z1-p4",
            "dst": "x0-y0-z0-p5",
            "speed_gbps": 100,
        }
        # There should be a pair of links in each direction between 2 nodes.
        self.assertIn(X_LINK_1, model["links"])
        self.assertIn(X_LINK_2, model["links"])
        self.assertIn(Y_LINK_1, model["links"])
        self.assertIn(Y_LINK_2, model["links"])
        self.assertIn(Z_LINK_1, model["links"])
        self.assertIn(Z_LINK_2, model["links"])

        # Verify that all x-dimension wrap-around links are present.
        for y in range(4):
            for z in range(4):
                x_wrap_link = {
                    "name": f"x3-y{y}-z{z}-p1:x0-y{y}-z{z}-p0",
                    "src": f"x3-y{y}-z{z}-p1",
                    "dst": f"x0-y{y}-z{z}-p0",
                    "speed_gbps": 100,
                }
                self.assertIn(x_wrap_link, model["links"])
        # Verify that all y-dimension wrap-around links are present.
        for x in range(4):
            for z in range(4):
                y_wrap_link = {
                    "name": f"x{x}-y0-z{z}-p2:x{x}-y3-z{z}-p3",
                    "src": f"x{x}-y0-z{z}-p2",
                    "dst": f"x{x}-y3-z{z}-p3",
                    "speed_gbps": 100,
                }
                self.assertIn(y_wrap_link, model["links"])
        # Verify that all z-dimension wrap-around links are present.
        for x in range(4):
            for y in range(4):
                z_wrap_link = {
                    "name": f"x{x}-y{y}-z0-p4:x{x}-y{y}-z3-p5",
                    "src": f"x{x}-y{y}-z0-p4",
                    "dst": f"x{x}-y{y}-z3-p5",
                    "speed_gbps": 100,
                }
                self.assertIn(z_wrap_link, model["links"])

        for link in model["links"]:
            self.assertEqual(link["speed_gbps"], 100)
            self.assertEqual(link["name"], f"{link['src']}:{link['dst']}")

    def test_t3d_bad(self):
        """
        Test building a bad 4x4x4 3D torus model. Omit tests for bad 3D mesh model because
        it works similarly.
        """
        GOOD_DIM = (4, 4, 4)
        GOOD_XPU = 1
        GOOD_PORT = 6
        GOOD_SPEED = 100

        # Dimension must match topology type.
        self.assertRaises(
            ValueError,
            build,
            TopoType.T3D_NT,
            "test",
            (4, 4),
            GOOD_XPU,
            GOOD_PORT,
            GOOD_SPEED,
        )

        # Dimension should not contain non-positive values or fractions.
        self.assertRaises(
            ValueError,
            build,
            TopoType.T3D_NT,
            "test",
            (0, 0, 0),
            GOOD_XPU,
            GOOD_PORT,
            GOOD_SPEED,
        )
        self.assertRaises(
            ValueError,
            build,
            TopoType.T3D_NT,
            "test",
            (4, -1, 4),
            GOOD_XPU,
            GOOD_PORT,
            GOOD_SPEED,
        )
        self.assertRaises(
            ValueError,
            build,
            TopoType.T3D_NT,
            "test",
            (4, 4.1, 4),
            GOOD_XPU,
            GOOD_PORT,
            GOOD_SPEED,
        )

        # xpu_per_node should not be anything other than 1.
        self.assertRaises(
            ValueError, build, TopoType.T3D_NT, "test", GOOD_DIM, 0, GOOD_PORT, GOOD_SPEED
        )
        self.assertRaises(
            ValueError, build, TopoType.T3D_NT, "test", GOOD_DIM, 2, GOOD_PORT, GOOD_SPEED
        )

        # port_per_node should not be anything other than 6.
        self.assertRaises(
            ValueError, build, TopoType.T3D_NT, "test", GOOD_DIM, GOOD_XPU, 3, GOOD_SPEED
        )
        self.assertRaises(
            ValueError, build, TopoType.T3D_NT, "test", GOOD_DIM, GOOD_XPU, 0, GOOD_SPEED
        )

        # port_speed_gbps must be positive.
        self.assertRaises(
            ValueError, build, TopoType.T3D_NT, "test", GOOD_DIM, GOOD_XPU, GOOD_PORT, -1
        )

    def test_clos_1tier_spec(self):
        """
        Test properties of a 1-tier Clos model.
        """
        model = build(
            topo=TopoType.CLOS,
            name="test",
            dimension=(4, 4),
            xpu_per_node=4,
            port_per_node=4,
            port_speed_gbps=100,
            t1_reserved=0,
        )
        self.assertIsNotNone(model)
        self.assertEqual(model["name"], "test")
        self.assertEqual(model["topology"], "CLOS")
        self.assertEqual(model["dimx"], 4)
        self.assertEqual(model["dimy"], 4)
        self.assertEqual(model["dimz"], 0)
        self.assertEqual(model["total_nodes"], 4)
        self.assertEqual(model["num_pods"], 1)
        self.assertEqual(len(model["nodes"]), 4)
        for node in model["nodes"]:
            coord = node["coordinates"]
            self.assertEqual(node["name"], f"pod{coord[0]}-n{coord[1]}")
            self.assertEqual(node["num_xpu"], 4)
            self.assertEqual(len(node["ports"]), 4)
            for port in node["ports"]:
                self.assertLess(port["index"], 4)
                self.assertEqual(port["name"], f"{node['name']}-p{port['index']}")
                self.assertEqual(port["speed_gbps"], 100)
        self.assertEqual(len(model["tier0"]), 4)
        for t0 in model["tier0"]:
            self.assertEqual(t0["tier"], 0)
            coord = t0["coordinates"]
            self.assertEqual(t0["name"], f"pod{coord[0]}-t{coord[1]}")
            self.assertEqual(len(t0["ports"]), 8)
            for port in t0["ports"]:
                self.assertLess(port["index"], 8)
                self.assertEqual(port["name"], f"{t0['name']}-p{port['index']}")
                self.assertEqual(port["speed_gbps"], 100)
        self.assertEqual(len(model["tier1"]), 0)
        N0_T0_LINK_1 = {
            "name": "pod0-n0-p0:pod0-t0-p0",
            "src": "pod0-n0-p0",
            "dst": "pod0-t0-p0",
            "speed_gbps": 100,
        }
        N0_T0_LINK_2 = {
            "name": "pod0-t0-p0:pod0-n0-p0",
            "src": "pod0-t0-p0",
            "dst": "pod0-n0-p0",
            "speed_gbps": 100,
        }
        N1_T3_LINK_1 = {
            "name": "pod0-n1-p3:pod0-t3-p2",
            "src": "pod0-n1-p3",
            "dst": "pod0-t3-p2",
            "speed_gbps": 100,
        }
        N1_T3_LINK_2 = {
            "name": "pod0-t3-p2:pod0-n1-p3",
            "src": "pod0-t3-p2",
            "dst": "pod0-n1-p3",
            "speed_gbps": 100,
        }
        # Links of both directions between node and tier-0 should exist.
        self.assertIn(N0_T0_LINK_1, model["links"])
        self.assertIn(N0_T0_LINK_2, model["links"])
        self.assertIn(N1_T3_LINK_1, model["links"])
        self.assertIn(N1_T3_LINK_2, model["links"])

    def test_clos_2tier_spec(self):
        """
        Test properties of a 2-tier Clos model.
        """
        model = build(
            topo=TopoType.CLOS,
            name="test",
            dimension=(4, 4, 4),
            xpu_per_node=4,
            port_per_node=4,
            port_speed_gbps=100,
            t1_reserved=0,
        )
        self.assertIsNotNone(model)
        self.assertEqual(model["name"], "test")
        self.assertEqual(model["topology"], "CLOS")
        self.assertEqual(model["dimx"], 4)
        self.assertEqual(model["dimy"], 4)
        self.assertEqual(model["dimz"], 4)
        self.assertEqual(model["total_nodes"], 8)
        self.assertEqual(model["num_pods"], 2)
        self.assertEqual(len(model["nodes"]), 8)
        for node in model["nodes"]:
            coord = node["coordinates"]
            self.assertEqual(node["name"], f"pod{coord[0]}-n{coord[1]}")
            self.assertEqual(node["num_xpu"], 4)
            self.assertEqual(len(node["ports"]), 4)
            for port in node["ports"]:
                self.assertLess(port["index"], 4)
                self.assertEqual(port["name"], f"{node['name']}-p{port['index']}")
                self.assertEqual(port["speed_gbps"], 100)
        self.assertEqual(len(model["tier0"]), 4 * model["num_pods"])
        for t0 in model["tier0"]:
            self.assertEqual(t0["tier"], 0)
            coord = t0["coordinates"]
            self.assertEqual(t0["name"], f"pod{coord[0]}-t{coord[1]}")
            self.assertEqual(len(t0["ports"]), 8)
            for port in t0["ports"]:
                self.assertLess(port["index"], 8)
                self.assertEqual(port["name"], f"{t0['name']}-p{port['index']}")
                self.assertEqual(port["speed_gbps"], 100)
        self.assertEqual(len(model["tier1"]), 4)
        for t1 in model["tier1"]:
            self.assertEqual(t1["tier"], 1)
            coord = t1["coordinates"]
            self.assertEqual(t1["name"], f"s{coord[0]}")
            self.assertEqual(len(t1["ports"]), 8)
            for port in t1["ports"]:
                self.assertLess(port["index"], 8)
                self.assertEqual(port["name"], f"{t1['name']}-p{port['index']}")
                self.assertEqual(port["speed_gbps"], 100)
        # In every pod, links of both directions between node and tier-0 should exist.
        for i in range(model["num_pods"]):
            N0_T0_LINK_1 = {
                "name": f"pod{i}-n0-p0:pod{i}-t0-p0",
                "src": f"pod{i}-n0-p0",
                "dst": f"pod{i}-t0-p0",
                "speed_gbps": 100,
            }
            N0_T0_LINK_2 = {
                "name": f"pod{i}-t0-p0:pod{i}-n0-p0",
                "src": f"pod{i}-t0-p0",
                "dst": f"pod{i}-n0-p0",
                "speed_gbps": 100,
            }
            N1_T3_LINK_1 = {
                "name": f"pod{i}-n1-p3:pod{i}-t3-p2",
                "src": f"pod{i}-n1-p3",
                "dst": f"pod{i}-t3-p2",
                "speed_gbps": 100,
            }
            N1_T3_LINK_2 = {
                "name": f"pod{i}-t3-p2:pod{i}-n1-p3",
                "src": f"pod{i}-t3-p2",
                "dst": f"pod{i}-n1-p3",
                "speed_gbps": 100,
            }
            self.assertIn(N0_T0_LINK_1, model["links"])
            self.assertIn(N0_T0_LINK_2, model["links"])
            self.assertIn(N1_T3_LINK_1, model["links"])
            self.assertIn(N1_T3_LINK_2, model["links"])
        T0_S0_LINK_1 = {
            "name": "pod0-t0-p1:s0-p0",
            "src": "pod0-t0-p1",
            "dst": "s0-p0",
            "speed_gbps": 100,
        }
        T0_S0_LINK_2 = {
            "name": "s0-p0:pod0-t0-p1",
            "src": "s0-p0",
            "dst": "pod0-t0-p1",
            "speed_gbps": 100,
        }
        T3_S3_LINK_1 = {
            "name": "pod1-t3-p7:s3-p7",
            "src": "pod1-t3-p7",
            "dst": "s3-p7",
            "speed_gbps": 100,
        }
        T3_S3_LINK_2 = {
            "name": "s3-p7:pod1-t3-p7",
            "src": "s3-p7",
            "dst": "pod1-t3-p7",
            "speed_gbps": 100,
        }
        # Links between tier-0 and tier-1 should exist.
        self.assertIn(T0_S0_LINK_1, model["links"])
        self.assertIn(T0_S0_LINK_2, model["links"])
        self.assertIn(T3_S3_LINK_1, model["links"])
        self.assertIn(T3_S3_LINK_2, model["links"])

    def test_clos_bad(self):
        """
        Test building a bad Clos model.
        """
        GOOD_DIM = (4, 4, 4)
        GOOD_XPU = 4
        GOOD_PORT = 4
        GOOD_SPEED = 100
        NO_OUTPUT = None
        GOOD_T1_RESERVE = 0

        # Dimension must be 1-tier or 2-tier.
        self.assertRaises(
            NotImplementedError,
            build,
            TopoType.CLOS,
            "test",
            (4, 4, 4, 4),
            GOOD_XPU,
            GOOD_PORT,
            GOOD_SPEED,
            NO_OUTPUT,
            GOOD_T1_RESERVE,
        )

        # Dimension should not contain non-positive values or fractions.
        self.assertRaises(
            ValueError,
            build,
            TopoType.CLOS,
            "test",
            (0, 0, 0),
            GOOD_XPU,
            GOOD_PORT,
            GOOD_SPEED,
            NO_OUTPUT,
            GOOD_T1_RESERVE,
        )
        self.assertRaises(
            ValueError,
            build,
            TopoType.CLOS,
            "test",
            (4, -1, 4),
            GOOD_XPU,
            GOOD_PORT,
            GOOD_SPEED,
            NO_OUTPUT,
            GOOD_T1_RESERVE,
        )
        self.assertRaises(
            ValueError,
            build,
            TopoType.CLOS,
            "test",
            (4, 4.1, 4),
            GOOD_XPU,
            GOOD_PORT,
            GOOD_SPEED,
            NO_OUTPUT,
            GOOD_T1_RESERVE,
        )

        # xpu_per_node should not be at least 1.
        self.assertRaises(
            ValueError,
            build,
            TopoType.CLOS,
            "test",
            GOOD_DIM,
            0,
            GOOD_PORT,
            GOOD_SPEED,
            NO_OUTPUT,
            GOOD_T1_RESERVE,
        )

        # port_per_node should be at least 1.
        self.assertRaises(
            ValueError,
            build,
            TopoType.CLOS,
            "test",
            GOOD_DIM,
            GOOD_XPU,
            0,
            GOOD_SPEED,
            NO_OUTPUT,
            GOOD_T1_RESERVE,
        )

        # port_speed_gbps must be positive.
        self.assertRaises(
            ValueError,
            build,
            TopoType.CLOS,
            "test",
            GOOD_DIM,
            GOOD_XPU,
            GOOD_PORT,
            -1,
            NO_OUTPUT,
            GOOD_T1_RESERVE,
        )

        # T1 reserved ports should not cause num_pod to be fractional.
        self.assertRaises(
            ValueError,
            build,
            TopoType.CLOS,
            "test",
            GOOD_DIM,
            GOOD_XPU,
            GOOD_PORT,
            GOOD_SPEED,
            NO_OUTPUT,
            2,
        )


class TestClusterConstruction(unittest.TestCase):

    def setUp(self):
        self.env = simpy.Environment()

    def test_t2d(self):
        """
        Test properties of a 4x4 2D torus cluster.
        """
        model = build(
            topo=TopoType.T2D,
            name="test",
            dimension=(4, 4),
            xpu_per_node=1,
            port_per_node=4,
            port_speed_gbps=100,
        )
        cluster = Cluster(self.env, spec=model)
        self.assertIsNotNone(model)
        self.assertIsNotNone(cluster)
        self.assertEqual(cluster.name, "test")
        self.assertEqual(cluster.topo, TopoType.T2D)
        self.assertEqual(cluster.dimx, 4)
        self.assertEqual(cluster.dimy, 4)
        self.assertEqual(cluster.dimz, 0)
        self.assertEqual(cluster.numNodes(), 4 * 4)
        for node in cluster.allNodes().values():
            self.assertIsNone(node.pod_id)
            self.assertIsNone(node.node_id)
            self.assertLess(node.dimx, 4)
            self.assertLess(node.dimy, 4)
            self.assertEqual(node.dimz, None)
            self.assertEqual(node.numXPU(), 1)
            self.assertEqual(node.numIdleXPU(), 1)
            self.assertEqual(len(node._ports), 4)
            self.assertEqual(cluster.getIdleXPU(node.name), 1)

        NODE1_DIM = (2, 3)
        NODE2_DIM = (2, 0)
        NODE3_DIM = (1, 1)
        NODE4_DIM = (7, 9)
        self.assertTrue(cluster.hasNode(f"x{NODE1_DIM[0]}-y{NODE1_DIM[1]}"))
        self.assertTrue(cluster.hasNode(f"x{NODE2_DIM[0]}-y{NODE2_DIM[1]}"))
        self.assertTrue(cluster.hasNode(f"x{NODE3_DIM[0]}-y{NODE3_DIM[1]}"))
        self.assertFalse(cluster.hasNode(f"x{NODE4_DIM[0]}-y{NODE4_DIM[1]}"))
        node1 = cluster.getNodeByName(f"x{NODE1_DIM[0]}-y{NODE1_DIM[1]}")
        # 2D torus should have None 1 for z dimension.
        self.assertEqual(node1.dimx, NODE1_DIM[0])
        self.assertEqual(node1.dimy, NODE1_DIM[1])
        self.assertEqual(node1.dimz, None)
        # Node in mesh/torus has no pod or node ID.
        self.assertIsNone(node1.pod_id)
        self.assertIsNone(node1.node_id)
        node2 = cluster.getNodeByName(f"x{NODE2_DIM[0]}-y{NODE2_DIM[1]}")
        self.assertEqual(node2.dimx, NODE2_DIM[0])
        self.assertEqual(node2.dimy, NODE2_DIM[1])
        self.assertEqual(node2.dimz, None)
        self.assertIsNone(node2.pod_id)
        self.assertIsNone(node2.node_id)
        node3 = cluster.getNodeByName(f"x{NODE3_DIM[0]}-y{NODE3_DIM[1]}")
        self.assertEqual(node3.dimx, NODE3_DIM[0])
        self.assertEqual(node3.dimy, NODE3_DIM[1])
        self.assertEqual(node3.dimz, None)
        self.assertIsNone(node3.pod_id)
        self.assertIsNone(node3.node_id)
        # There should only be one link from node1 to node2, i.e., wrap-around link.
        links = cluster.findLinksBetweenNodes(node1, node2)
        self.assertEqual(len(links), 1)
        src_port_name = "x2-y3-p3"
        dst_port_name = "x2-y0-p2"
        self.assertTrue(cluster.hasPort(src_port_name))
        self.assertTrue(cluster.hasPort(dst_port_name))
        self.assertEqual(links[0].src_port.name, src_port_name)
        self.assertEqual(links[0].dst_port.name, dst_port_name)
        self.assertEqual(cluster.findPeerPortOfPort(src_port_name).name, dst_port_name)
        self.assertEqual(cluster.findPeerPortOfPort(dst_port_name).name, src_port_name)
        self.assertEqual(links[0].speed_gbps, 100)
        # There should be no link from node1 to node3 (it's diagonal).
        self.assertEqual(cluster.findLinksBetweenNodes(node1, node3), [])

    def test_t3d(self):
        """
        Test properties of a 4x4x4 3D torus model.
        """
        model = build(
            topo=TopoType.T3D_NT,
            name="test",
            dimension=(4, 4, 4),
            xpu_per_node=1,
            port_per_node=6,
            port_speed_gbps=100,
        )
        cluster = Cluster(self.env, spec=model)
        self.assertIsNotNone(model)
        self.assertIsNotNone(cluster)
        self.assertEqual(cluster.name, "test")
        self.assertEqual(cluster.topo, TopoType.T3D_NT)
        self.assertEqual(cluster.dimx, 4)
        self.assertEqual(cluster.dimy, 4)
        self.assertEqual(cluster.dimz, 4)
        self.assertEqual(cluster.numNodes(), 4 * 4 * 4)
        for node in cluster.allNodes().values():
            self.assertIsNone(node.pod_id)
            self.assertIsNone(node.node_id)
            self.assertLess(node.dimx, 4)
            self.assertLess(node.dimy, 4)
            self.assertLess(node.dimz, 4)
            self.assertEqual(node.numXPU(), 1)
            self.assertEqual(node.numIdleXPU(), 1)
            self.assertEqual(len(node._ports), 6)
            self.assertEqual(cluster.getIdleXPU(node.name), 1)

        NODE1_DIM = (0, 0, 2)
        NODE2_DIM = (3, 0, 2)
        NODE3_DIM = (1, 1, 3)
        NODE4_DIM = (5, 6, 7)
        self.assertTrue(
            cluster.hasNode(f"x{NODE1_DIM[0]}-y{NODE1_DIM[1]}-z{NODE1_DIM[2]}")
        )
        self.assertTrue(
            cluster.hasNode(f"x{NODE2_DIM[0]}-y{NODE2_DIM[1]}-z{NODE2_DIM[2]}")
        )
        self.assertTrue(
            cluster.hasNode(f"x{NODE3_DIM[0]}-y{NODE3_DIM[1]}-z{NODE3_DIM[2]}")
        )
        self.assertFalse(
            cluster.hasNode(f"x{NODE4_DIM[0]}-y{NODE4_DIM[1]}-z{NODE4_DIM[2]}")
        )
        node1 = cluster.getNodeByName(f"x{NODE1_DIM[0]}-y{NODE1_DIM[1]}-z{NODE1_DIM[2]}")
        self.assertEqual(node1.dimx, NODE1_DIM[0])
        self.assertEqual(node1.dimy, NODE1_DIM[1])
        self.assertEqual(node1.dimz, NODE1_DIM[2])
        # Node in mesh/torus has no pod or node ID.
        self.assertIsNone(node1.pod_id)
        self.assertIsNone(node1.node_id)
        node2 = cluster.getNodeByName(f"x{NODE2_DIM[0]}-y{NODE2_DIM[1]}-z{NODE2_DIM[2]}")
        self.assertEqual(node2.dimx, NODE2_DIM[0])
        self.assertEqual(node2.dimy, NODE2_DIM[1])
        self.assertEqual(node2.dimz, NODE2_DIM[2])
        self.assertIsNone(node2.pod_id)
        self.assertIsNone(node2.node_id)
        node3 = cluster.getNodeByName(f"x{NODE3_DIM[0]}-y{NODE3_DIM[1]}-z{NODE3_DIM[2]}")
        self.assertEqual(node3.dimx, NODE3_DIM[0])
        self.assertEqual(node3.dimy, NODE3_DIM[1])
        self.assertEqual(node3.dimz, NODE3_DIM[2])
        self.assertIsNone(node3.pod_id)
        self.assertIsNone(node3.node_id)
        # There should only be one link from node1 to node2, i.e., wrap-around link.
        links = cluster.findLinksBetweenNodes(node1, node2)
        self.assertEqual(len(links), 1)
        src_port_name = "x0-y0-z2-p0"
        dst_port_name = "x3-y0-z2-p1"
        self.assertTrue(cluster.hasPort(src_port_name))
        self.assertTrue(cluster.hasPort(dst_port_name))
        self.assertEqual(links[0].src_port.name, src_port_name)
        self.assertEqual(links[0].dst_port.name, dst_port_name)
        self.assertEqual(cluster.findPeerPortOfPort(src_port_name).name, dst_port_name)
        self.assertEqual(cluster.findPeerPortOfPort(dst_port_name).name, src_port_name)
        self.assertEqual(links[0].speed_gbps, 100)
        # There should be no link from node1 to node3 (it's diagonal).
        self.assertEqual(cluster.findLinksBetweenNodes(node1, node3), [])

    def test_clos_1tier(self):
        """
        Test properties of a 1-tier Clos model.
        """
        model = build(
            topo=TopoType.CLOS,
            name="test",
            dimension=(4, 4),
            xpu_per_node=4,
            port_per_node=4,
            port_speed_gbps=100,
            t1_reserved=0,
        )
        cluster = Cluster(self.env, spec=model)
        self.assertIsNotNone(model)
        self.assertIsNotNone(cluster)
        self.assertEqual(cluster.name, "test")
        self.assertEqual(cluster.topo, TopoType.CLOS)
        # 4 nodes and 4 T0 per pod, no T1.
        self.assertEqual(cluster.dimx, 4)
        self.assertEqual(cluster.dimy, 4)
        self.assertEqual(cluster.dimz, 0)
        self.assertEqual(cluster.num_pods, 1)
        self.assertEqual(cluster.numNodes(), 4)
        for node in cluster.allNodes().values():
            self.assertEqual(node.pod_id, 0)
            self.assertLess(node.node_id, 4)
            self.assertEqual(node.dimx, None)
            self.assertEqual(node.dimy, None)
            self.assertEqual(node.dimz, None)
            self.assertTrue(cluster.hasNodeInPod(node.name, 0))
            self.assertEqual(node.numXPU(), 4)
            self.assertEqual(node.numIdleXPU(), 4)
            self.assertEqual(len(node._ports), 4)
            self.assertEqual(cluster.getIdleXPU(node.name), 4)
            # There is only 1 pod, all nodes are in this pod.
            self.assertIn(node.name, cluster.pods[0])
        self.assertEqual(len(cluster.tier0s), 4)
        for t in cluster.tier0s.values():
            self.assertEqual(t.tier, 0)
            self.assertEqual(t.pod_id, 0)
            self.assertLess(t.switch_id, 4)
            self.assertEqual(len(t._ports), 8)
        # 1-tier Clos has no T1.
        self.assertEqual(cluster.tier1s, {})

        NODE1_DIM = (0, 0)
        NODE2_DIM = (0, 3)
        NODE3_DIM = (1, 1)
        T0_1_DIM = (0, 0)
        T0_2_DIM = (0, 1)
        self.assertTrue(cluster.hasNode(f"pod{NODE1_DIM[0]}-n{NODE1_DIM[1]}"))
        self.assertTrue(cluster.hasNode(f"pod{NODE2_DIM[0]}-n{NODE2_DIM[1]}"))
        self.assertFalse(cluster.hasNode(f"pod{NODE3_DIM[0]}-n{NODE3_DIM[1]}"))
        self.assertTrue(cluster.hasT0(f"pod{T0_1_DIM[0]}-t{T0_1_DIM[1]}"))
        self.assertTrue(cluster.hasT0(f"pod{T0_2_DIM[0]}-t{T0_2_DIM[1]}"))
        node1 = cluster.getNodeByName(f"pod{NODE1_DIM[0]}-n{NODE1_DIM[1]}")
        self.assertEqual(node1.pod_id, NODE1_DIM[0])
        self.assertEqual(node1.node_id, NODE1_DIM[1])
        node2 = cluster.getNodeByName(f"pod{NODE2_DIM[0]}-n{NODE2_DIM[1]}")
        self.assertEqual(node2.pod_id, NODE2_DIM[0])
        self.assertEqual(node2.node_id, NODE2_DIM[1])
        # There should be no link directly connecting nodes.
        self.assertEqual(cluster.findLinksBetweenNodes(node1, node2), [])
        t0_1_obj = cluster.getT0ByName(f"pod{T0_1_DIM[0]}-t{T0_1_DIM[1]}")
        self.assertIsNotNone(t0_1_obj)
        # There is 1 link from n0 to t0.
        links = cluster.findLinksBetweenNodes(node1, t0_1_obj)
        self.assertEqual(len(links), 1)
        src_port_name = "pod0-n0-p0"
        dst_port_name = "pod0-t0-p0"
        self.assertTrue(cluster.hasPort(src_port_name))
        self.assertTrue(cluster.hasPort(dst_port_name))
        self.assertEqual(links[0].src_port.name, src_port_name)
        self.assertEqual(links[0].dst_port.name, dst_port_name)
        self.assertEqual(cluster.findPeerPortOfPort(src_port_name).name, dst_port_name)
        self.assertEqual(cluster.findPeerPortOfPort(dst_port_name).name, src_port_name)
        self.assertEqual(links[0].speed_gbps, 100)
        # There is 1 link from n3 to t1.
        t0_2_obj = cluster.getT0ByName(f"pod{T0_2_DIM[0]}-t{T0_2_DIM[1]}")
        self.assertIsNotNone(t0_2_obj)
        links = cluster.findLinksBetweenNodes(node2, t0_2_obj)
        self.assertEqual(len(links), 1)
        src_port_name = "pod0-n3-p1"
        dst_port_name = "pod0-t1-p6"
        self.assertTrue(cluster.hasPort(src_port_name))
        self.assertTrue(cluster.hasPort(dst_port_name))
        self.assertEqual(links[0].src_port.name, src_port_name)
        self.assertEqual(links[0].dst_port.name, dst_port_name)
        self.assertEqual(cluster.findPeerPortOfPort(src_port_name).name, dst_port_name)
        self.assertEqual(cluster.findPeerPortOfPort(dst_port_name).name, src_port_name)
        self.assertEqual(links[0].speed_gbps, 100)

    def test_clos_2tier(self):
        """
        Test properties of a 2-tier Clos model.
        """
        model = build(
            topo=TopoType.CLOS,
            name="test",
            dimension=(4, 4, 4),
            xpu_per_node=4,
            port_per_node=4,
            port_speed_gbps=100,
            t1_reserved=0,
        )
        cluster = Cluster(self.env, spec=model)
        self.assertIsNotNone(model)
        self.assertIsNotNone(cluster)
        self.assertEqual(cluster.name, "test")
        self.assertEqual(cluster.topo, TopoType.CLOS)
        # 4 nodes and 4 T0 per pod, 4 T1.
        self.assertEqual(cluster.dimx, 4)
        self.assertEqual(cluster.dimy, 4)
        self.assertEqual(cluster.dimz, 4)
        self.assertEqual(cluster.num_pods, 2)
        self.assertEqual(cluster.numNodes(), 8)
        for node in cluster.allNodes().values():
            self.assertLess(node.pod_id, 2)
            self.assertLess(node.node_id, 4)
            self.assertEqual(node.dimx, None)
            self.assertEqual(node.dimy, None)
            self.assertEqual(node.dimz, None)
            self.assertEqual(node.numXPU(), 4)
            self.assertEqual(node.numIdleXPU(), 4)
            self.assertEqual(len(node._ports), 4)
            self.assertEqual(cluster.getIdleXPU(node.name), 4)
            self.assertIn(node.name, cluster.pods[node.pod_id])
        self.assertEqual(len(cluster.tier0s), 4 * 2)
        for t in cluster.tier0s.values():
            self.assertEqual(t.tier, 0)
            self.assertLess(t.pod_id, 2)
            self.assertLess(t.switch_id, 4)
            self.assertEqual(len(t._ports), 8)
        self.assertEqual(len(cluster.tier1s), 4)
        for s in cluster.tier1s.values():
            self.assertEqual(s.tier, 1)
            # Tier 1 switch is global, has no pod id.
            self.assertIsNone(s.pod_id)
            self.assertLess(t.switch_id, 4)
            self.assertEqual(len(t._ports), 8)

        NODE1_DIM = (0, 0)
        NODE2_DIM = (1, 3)
        NODE3_DIM = (2, 1)
        T0_DIM = (0, 0)
        T1_DIM = (1,)
        self.assertTrue(cluster.hasNode(f"pod{NODE1_DIM[0]}-n{NODE1_DIM[1]}"))
        self.assertTrue(cluster.hasNode(f"pod{NODE2_DIM[0]}-n{NODE2_DIM[1]}"))
        self.assertFalse(cluster.hasNode(f"pod{NODE3_DIM[0]}-n{NODE3_DIM[1]}"))
        self.assertTrue(cluster.hasT0(f"pod{T0_DIM[0]}-t{T0_DIM[1]}"))
        self.assertTrue(cluster.hasT1(f"s{T1_DIM[0]}"))
        node1 = cluster.getNodeByName(f"pod{NODE1_DIM[0]}-n{NODE1_DIM[1]}")
        self.assertEqual(node1.pod_id, NODE1_DIM[0])
        self.assertEqual(node1.node_id, NODE1_DIM[1])
        node2 = cluster.getNodeByName(f"pod{NODE2_DIM[0]}-n{NODE2_DIM[1]}")
        self.assertEqual(node2.pod_id, NODE2_DIM[0])
        self.assertEqual(node2.node_id, NODE2_DIM[1])
        # There should be no link directly connecting nodes.
        self.assertEqual(cluster.findLinksBetweenNodes(node1, node2), [])
        t0_obj = cluster.getT0ByName(f"pod{T0_DIM[0]}-t{T0_DIM[1]}")
        self.assertIsNotNone(t0_obj)
        # There is 1 link from n0 to t0.
        links = cluster.findLinksBetweenNodes(node1, t0_obj)
        self.assertEqual(len(links), 1)
        src_port_name = "pod0-n0-p0"
        dst_port_name = "pod0-t0-p0"
        self.assertTrue(cluster.hasPort(src_port_name))
        self.assertTrue(cluster.hasPort(dst_port_name))
        self.assertEqual(links[0].src_port.name, src_port_name)
        self.assertEqual(links[0].dst_port.name, dst_port_name)
        self.assertEqual(cluster.findPeerPortOfPort(src_port_name).name, dst_port_name)
        self.assertEqual(cluster.findPeerPortOfPort(dst_port_name).name, src_port_name)
        self.assertEqual(links[0].speed_gbps, 100)
        # There is 1 link from t0 to s1.
        t1_obj = cluster.getT1ByName(f"s{T1_DIM[0]}")
        self.assertIsNotNone(t1_obj)
        links = cluster.findLinksBetweenNodes(t0_obj, t1_obj)
        self.assertEqual(len(links), 1)
        src_port_name = "pod0-t0-p3"
        dst_port_name = "s1-p0"
        self.assertTrue(cluster.hasPort(src_port_name))
        self.assertTrue(cluster.hasPort(dst_port_name))
        self.assertEqual(links[0].src_port.name, src_port_name)
        self.assertEqual(links[0].dst_port.name, dst_port_name)
        self.assertEqual(cluster.findPeerPortOfPort(src_port_name).name, dst_port_name)
        self.assertEqual(cluster.findPeerPortOfPort(dst_port_name).name, src_port_name)
        self.assertEqual(links[0].speed_gbps, 100)
