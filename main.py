import cv2
from pathlib import Path
from process import *
import process.feature_extraction as fe
import process.feature_matching as fm
from util import *
import yaml
import shutil


def main():
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
    output_suffix = (
        max([int(p.stem.split("_")[-1]) for p in output_path.glob("recon_*")]) + 1
        if output_path.exists() and len(list(output_path.glob("recon_*"))) > 0
        else 1
    )
    output_path = output_path / f"recon_{output_suffix}"
    img_path = output_path / path_config["img_path"]
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

    # 保存配置文件
    with open(output_path / "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    # 检查num_images配置
    assert (
        spec_config["num_images"] >= 5 or spec_config["num_images"] == -1
    ), "num_images配置必须 >=5 或 =-1"

    # 打印输出信息
    print(f"=" * 40)
    print(f"AI4701 室内场景三维重建")
    print(f"输出路径: {output_path}")
    print(
        f"重建照片数量: {spec_config['num_images'] if spec_config['num_images'] > 0 else '全部'}"
    )
    print(f"=" * 40)

    try:
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

        # 初始化图像mask
        images_mask = np.ones(len(images), dtype=bool)
        if spec_config["num_images"] > 0:
            images_mask[: spec_config["num_images"]] = True
            images_mask[spec_config["num_images"] :] = False
        elif spec_config["num_images"] == -1:
            images_mask[:] = True

        # 应用mask过滤图像
        images = [img for i, img in enumerate(images) if images_mask[i]]

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
                img,
                features[i][0],
                None,
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
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
        points3D, colors3D, cameras[0], cameras[1] = cache_wrapper(
            step_config["cache_key"], step_config["regenerate"], init_recon
        )(
            features1=features[0],
            features2=features[1],
            camera1=cameras[0],
            camera2=cameras[1],
            match=all_matches[(0, 1)],
            img1=images[0],
            img2=images[1],
            max_depth=spec_config["max_depth"],
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
        # Step4: 场景重建 (PnP/对极几何) + Bundle Adjustment优化
        # ===============================================================
        step_config = next(steps_config)
        recon_method = spec_config["recon_method"]
        cache_key = step_config["cache_key"] + f"_{recon_method}"
        print(f"\n处理: {step_config['desc']}")
        print(f"重建方法: {recon_method}")

        if spec_config["skip_BA"]:
            print("跳过BA优化")
        # 执行PnP或对极几何算法来重建场景，可选是否进行BA
        cameras_recon_all, points3D_recon_all, colors3D_recon_all = cache_wrapper(
            cache_key, step_config["regenerate"], recon_all
        )(
            recon_method,
            points3D,
            features,
            cameras,
            all_matches,
            spec_config["max_depth"],
            images,
            spec_config["skip_BA"],
        )

        print(f"点云数量: {len(points3D_recon_all)}")

        # 测试：展示重建结果
        camera_poses_recon_all = [camera.get_pose() for camera in cameras_recon_all[1:]]
        visualize_camera_pose_and_pcd(
            camera_poses_recon_all,
            points3D_recon_all,
            colors3D_recon_all,
            pcd_visual_without_cameras_param_path,
            pcd_visual_with_cameras_param_path,
            show=step_config["show_pcd"],
            save=step_config["save_pcd"],
            title="2_Recon_all_Result",
            save_dir=img_path_pcd,
        )

        # 保存重建的点云和相机坐标系
        pcd = create_point_cloud(points3D_recon_all, colors3D_recon_all)
        save_point_cloud(pcd, output_path / "pcd.ply")
        extrinsic_matrices_list = [
            camera.get_flattened_extrinsic() for camera in cameras_recon_all
        ]
        np.savetxt(output_path / "extrinsics.txt", np.array(extrinsic_matrices_list))
    except Exception as e:
        shutil.rmtree(output_path, ignore_errors=True)
        raise e


if __name__ == "__main__":
    main()
