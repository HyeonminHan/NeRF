import open3d as o3d

pc = o3d.io.read_point_cloud('../pcd/incre_test.pcd')

o3d.visualization.draw_geometries([pc])