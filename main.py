import cv2
from pathlib import Path
from process import *
import process.feature_extraction as fe
import process.feature_matching as fm
from util import *
import yaml


def main():
    print(f"=" * 40)
    print(f"AI4701 室内场景三维重建")
    print(f"详细配置见 config/config.yaml")
    print(f"=" * 40)

    # 导入配置文件
    with open(Path(__file__).parent / "config" / "config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 获取配置
    path_config = config["path"]
    spec_config = config["spec_config"]
    steps_config = iter(config["steps"])

    # 读取数据/保存路径
    data_path = Path(path_config["data_path"])
    output_path = Path(path_config["output_path"])
    img_path = Path(path_config["img_path"])
    img_path_pcd = img_path / "PCD"
    pcd_visual_without_cameras_param_path = Path(
        path_config["pcd_visual_without_cameras_param_path"]
    )
    pcd_visual_with_cameras_param_path = Path(
        path_config["pcd_visual_with_cameras_param_path"]
    )

    # 创建输出目录
    output_path.mkdir(parents=True, exist_ok=True)
    img_path.mkdir(parents=True, exist_ok=True)
    img_path_pcd.mkdir(parents=True, exist_ok=True)

    # ===============================================================
    # Step0: 导入图像及相机参数
    # ===============================================================
    step_config = next(steps_config)
    print(f"\n处理: {step_config['desc']}")

    # 加载图像和相机内参
    images, camera_intrinsic = cache_wrapper(
        step_config["cache_key"],
        step_config["regenerate"],
        load_images_and_camera_intrinsic,
    )(data_path)

    # 测试：展示图像和相机参数
    for i, img in enumerate(images[:2]):
        visualize_img(
            img,
            f"Image {i+1}",
            show=step_config["show_img"],
            save=step_config["save_img"],
            save_dir=img_path,
        )
    print("相机内参矩阵:")
    print(camera_intrinsic)

    # ===============================================================
    # Step1: 图像特征提取 (SIFT)
    # ===============================================================
    step_config = next(steps_config)
    print(f"\n处理: {step_config['desc']}")

    # 运行SIFT特征提取
    features = cache_wrapper(
        step_config["cache_key"],
        step_config["regenerate"],
        extract_features,
        fe.after_load,
    )(images)
    print(f"平均特征点数量: {np.mean([len(f[0]) for f in features]):.2f}")

    # 测试：展示图像特征点
    for i, img in enumerate(images[:2]):
        img_with_features = cv2.drawKeypoints(
            img, features[i][0], None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        visualize_img(
            img_with_features,
            f"Features in Image {i}",
            show=step_config["show_img"],
            save=step_config["save_img"],
            save_dir=img_path,
        )

    # ===============================================================
    # Step2: 图像特征匹配
    # ===============================================================
    step_config = next(steps_config)
    print(f"\n处理: {step_config['desc']}")

    # 运行特征匹配
    all_matches = cache_wrapper(
        step_config["cache_key"],
        step_config["regenerate"],
        match_all_pairs,
        fm.after_load,
    )(features)
    print(f"平均匹配点数量: {np.mean([len(m) for m in all_matches.values()]):.2f}")

    # 测试：展示匹配结果
    img1_indx = 0
    img2_indx = 3
    img_matches = visualize_matches(
        images[img1_indx],
        images[img2_indx],
        all_matches[(img1_indx, img2_indx)],
        features[img1_indx],
        features[img2_indx],
    )
    visualize_img(
        img_matches,
        f"Matches between Image {img1_indx} and Image {img2_indx}",
        show=step_config["show_img"],
        save=step_config["save_img"],
        save_dir=img_path,
    )

    # ===============================================================
    # Step3: 场景初始化 (对极几何)
    # ===============================================================
    step_config = next(steps_config)
    print(f"\n处理: {step_config['desc']}")

    # 初始化相机对象
    cameras = []
    for i in range(len(images)):
        camera = Camera(camera_intrinsic)
        cameras.append(camera)

    # 初始化重建
    points3D, colors3D = cache_wrapper(
        step_config["cache_key"], step_config["regenerate"], init_recon
    )(
        features1=features[0],
        features2=features[1],
        camera1=cameras[0],
        camera2=cameras[1],
        match=all_matches[(0, 1)],
        img1=images[0],
        img2=images[1],
    )

    # 测试：展示初始化结果
    print(f"初始化点云数量: {len(points3D)}")
    camera_poses_init = [cameras[1].get_pose()]
    visualize_camera_pose_and_pcd(
        camera_poses_init,
        points3D,
        colors3D,
        pcd_visual_without_cameras_param_path,
        pcd_visual_with_cameras_param_path,
        show=step_config["show_pcd"],
        save=step_config["save_pcd"],
        title="1_Init_Recon_Result",
        save_dir=img_path_pcd,
    )

    # ===============================================================
    # Step4: 场景重建 (PnP/对极几何)
    # ===============================================================
    step_config = next(steps_config)
    print(f"\n处理: {step_config['desc']}")

    # 执行PnP算法来估计相机姿态
    cameras_pnp, points3D_pnp, colors3D_pnp = cache_wrapper(
        step_config["cache_key"], step_config["regenerate"], pnp_recon
    )(points3D, features, cameras, all_matches, images)

    print(f"点云数量: {len(points3D_pnp)}")

    # 测试：展示重建结果
    camera_poses_pnp = [camera.get_pose() for camera in cameras_pnp[1:]]
    visualize_camera_pose_and_pcd(
        camera_poses_pnp,
        points3D_pnp,
        colors3D_pnp,
        pcd_visual_without_cameras_param_path,
        pcd_visual_with_cameras_param_path,
        show=step_config["show_pcd"],
        save=step_config["save_pcd"],
        title="2_PnP_Result",
        save_dir=img_path_pcd,
    )

    # ===============================================================
    # Step5: 场景优化 (Bundle Adjustment)
    # ===============================================================
    step_config = next(steps_config)
    print(f"\n处理: {step_config['desc']}")

    if spec_config["skip_ba"]:
        print("跳过Bundle Adjustment优化")
        return

    # 执行BA算法来优化相机姿态和点云
    optimized_points3D, optimized_cameras, optimized_colors = cache_wrapper(
        step_config["cache_key"],
        step_config["regenerate"],
        perform_BA,
    )(
        points3D=points3D_pnp,
        cameras=cameras_pnp,
        colors=colors3D_pnp,
        show=step_config["show_img"],
        save=step_config["save_img"],
        save_dir=img_path,
    )

    # 展示优化前后点云数量变化
    print(f"优化前点云数量: {points3D_pnp.shape[0]}")
    print(f"优化后点云数量: {optimized_points3D.shape[0]}")
    print(
        f"点云变化百分比: {100 * (optimized_points3D.shape[0] / points3D_pnp.shape[0]):.2f}%"
    )

    # 测试：展示优化结果
    # for i in range(1, len(optimized_cameras)):
    #     optimized_camera_poses = [optimized_cameras[i].get_pose()]
    #     visualize_camera_pose_and_pcd(
    #         optimized_camera_poses,
    #         optimized_points3D,
    #         optimized_colors,
    #         camera_param_path,
    #         show=step_config["show_pcd"],
    #         save=step_config["save_pcd"],
    #         title=f"3_BA_Result_{i}",
    #         save_dir=img_path_pcd,
    #     )
    optimized_camera_poses = [camera.get_pose() for camera in optimized_cameras[1:]]
    visualize_camera_pose_and_pcd(
        optimized_camera_poses,
        optimized_points3D,
        optimized_colors,
        pcd_visual_without_cameras_param_path,
        pcd_visual_with_cameras_param_path,
        show=step_config["show_pcd"],
        save=step_config["save_pcd"],
        title="3_BA_Result",
        save_dir=img_path_pcd,
    )

    # 保存重建的点云和相机坐标系
    pcd = create_point_cloud(optimized_points3D, optimized_colors)
    save_point_cloud(pcd, output_path / "pcd.ply")
    extrinsic_matrices_list = [
        camera.get_flattened_extrinsic() for camera in optimized_cameras
    ]
    np.savetxt(output_path / "extrinsics.txt", np.array(extrinsic_matrices_list))


if __name__ == "__main__":
    main()
