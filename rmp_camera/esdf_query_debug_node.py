import math

import rclpy
from geometry_msgs.msg import Point, Vector3
from nvblox_msgs.srv import EsdfAndGradients
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node


def parse_points(points_text):
    points = []
    for item in points_text.split(";"):
        item = item.strip()
        if not item:
            continue
        values = [float(value.strip()) for value in item.split(",")]
        if len(values) != 3:
            raise ValueError(f"Expected x,y,z point, got: {item}")
        points.append(values)
    return points


class EsdfQueryDebugNode(Node):
    def __init__(self):
        super().__init__("esdf_query_debug_node")
        self.declare_parameter("service_name", "/nvblox_node/get_esdf_and_gradient")
        self.declare_parameter("frame_id", "base_link")
        self.declare_parameter("points", "0.0,0.0,0.5;0.3,0.0,0.5;0.6,0.0,0.5")
        self.declare_parameter("aabb_size_m", 0.25)
        self.declare_parameter("repeat", False)
        self.declare_parameter("repeat_period_s", 1.0)

        self.service_name = self.get_parameter("service_name").value
        self.frame_id = self.get_parameter("frame_id").value
        self.points = parse_points(self.get_parameter("points").value)
        self.aabb_size_m = float(self.get_parameter("aabb_size_m").value)
        self.repeat = self._as_bool(self.get_parameter("repeat").value)
        self.repeat_period_s = float(self.get_parameter("repeat_period_s").value)

        self.client = self.create_client(EsdfAndGradients, self.service_name)
        self.pending = False
        self.query_index = 0
        self.timer = self.create_timer(0.2, self._tick)
        self.repeat_timer = None

        self.get_logger().info(
            f"Querying {self.service_name} in frame {self.frame_id} for {len(self.points)} points"
        )

    def _tick(self):
        if not self.client.service_is_ready():
            self.get_logger().info(
                f"Waiting for service {self.service_name}",
                throttle_duration_sec=2.0,
            )
            return
        if self.pending:
            return
        if self.query_index >= len(self.points):
            if self.repeat:
                self.query_index = 0
            else:
                self.get_logger().info("ESDF query debug complete")
                self.timer.cancel()
                return
        self._send_query(self.points[self.query_index])
        self.query_index += 1

    def _send_query(self, point):
        half = self.aabb_size_m * 0.5
        request = EsdfAndGradients.Request()
        request.update_esdf = True
        request.visualize_esdf = False
        request.use_aabb = True
        request.frame_id = self.frame_id
        request.aabb_min_m = Point(x=point[0] - half, y=point[1] - half, z=point[2] - half)
        request.aabb_size_m = Vector3(
            x=self.aabb_size_m,
            y=self.aabb_size_m,
            z=self.aabb_size_m,
        )
        future = self.client.call_async(request)
        future.add_done_callback(lambda done_future, p=point: self._handle_response(done_future, p))
        self.pending = True

    def _handle_response(self, future, point):
        self.pending = False
        try:
            response = future.result()
        except Exception as exc:
            self.get_logger().error(f"ESDF query failed for {point}: {exc}")
            return
        if not response.success:
            self.get_logger().warn(f"ESDF query returned success=false for {point}")
            return

        dims = response.esdf_and_gradients.layout.dim
        data = response.esdf_and_gradients.data
        if len(dims) < 3 or not data:
            self.get_logger().warn(f"ESDF query returned empty grid for {point}")
            return

        sx, sy, sz = int(dims[0].size), int(dims[1].size), int(dims[2].size)
        voxel = float(response.voxel_size_m)
        origin = [response.origin_m.x, response.origin_m.y, response.origin_m.z]
        ix = self._clamp(round((point[0] - origin[0] - 0.5 * voxel) / voxel), 0, sx - 1)
        iy = self._clamp(round((point[1] - origin[1] - 0.5 * voxel) / voxel), 0, sy - 1)
        iz = self._clamp(round((point[2] - origin[2] - 0.5 * voxel) / voxel), 0, sz - 1)

        distance = self._value(data, dims, ix, iy, iz)
        gradient = self._finite_difference_gradient(data, dims, ix, iy, iz, voxel)
        grad_norm = math.sqrt(sum(value * value for value in gradient if math.isfinite(value)))
        self.get_logger().info(
            "point=[%.3f, %.3f, %.3f] idx=[%d,%d,%d] distance=%.4f m gradient=[%.3f, %.3f, %.3f] |g|=%.3f"
            % (
                point[0],
                point[1],
                point[2],
                ix,
                iy,
                iz,
                distance,
                gradient[0],
                gradient[1],
                gradient[2],
                grad_norm,
            )
        )

    def _value(self, data, dims, ix, iy, iz):
        return float(data[ix * int(dims[1].stride) + iy * int(dims[2].stride) + iz])

    def _finite_difference_gradient(self, data, dims, ix, iy, iz, voxel):
        sx, sy, sz = int(dims[0].size), int(dims[1].size), int(dims[2].size)

        def diff(axis):
            indices_minus = [ix, iy, iz]
            indices_plus = [ix, iy, iz]
            if axis == 0:
                if ix <= 0 or ix >= sx - 1:
                    return float("nan")
                indices_minus[0] -= 1
                indices_plus[0] += 1
            elif axis == 1:
                if iy <= 0 or iy >= sy - 1:
                    return float("nan")
                indices_minus[1] -= 1
                indices_plus[1] += 1
            else:
                if iz <= 0 or iz >= sz - 1:
                    return float("nan")
                indices_minus[2] -= 1
                indices_plus[2] += 1
            minus = self._value(data, dims, *indices_minus)
            plus = self._value(data, dims, *indices_plus)
            if minus <= -999.0 or plus <= -999.0:
                return float("nan")
            return (plus - minus) / (2.0 * voxel)

        return [diff(0), diff(1), diff(2)]

    def _clamp(self, value, low, high):
        return max(low, min(high, int(value)))

    def _as_bool(self, value):
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in ("1", "true", "yes", "on")


def main(args=None):
    rclpy.init(args=args)
    node = EsdfQueryDebugNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
