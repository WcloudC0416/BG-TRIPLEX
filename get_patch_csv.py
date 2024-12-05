import csv
import os
import pyvips

radius_per_patch = 224          # 一个patch的大小 (224, 224)
num_neighbors = 5               # 邻域大小，5*5
n_radius = radius_per_patch * num_neighbors     # 领域半径
### 生成image和mask的patch
choice = input("1.her2st\n2.stnet\n3.skin\n4.visium")
if choice == 1:
    patient = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    for i in patient:
        if i == 'A' or i == 'B' or i == 'C' or i == 'D':
            for j in range(6):
                testset = f'data/her2st/ST-imgs/{i}/{i + str(j+1)}'
                outdir = 'data/her2st'
                fig_name = os.listdir(testset)
                for h in range(3):
                    if fig_name[h].endswith("cat_image.jpg" or "seg.jpg"):
                        continue
                    else:
                        figname = fig_name[h]
                        break
                image_file_path = testset + '/' + figname
                tsv_file_path = f'{outdir}/ST-spotfiles/{i + str(j+1)}_selection.tsv'
                nuclei_file_path = f'{testset}/seg.jpg'
                edge_file_path = f'{testset}/cat_image.jpg'
                per_patch_out_dir = f'{outdir}/gen_per_patch'
                n_patches_out_dir = f'{outdir}/gen_n_patches'
                nuc_per_patch_out_dir = f'{outdir}/gen_nuc_per_patch'
                nuc_n_patches_out_dir = f'{outdir}/gen_nuc_n_patches'
                edge_per_patch_out_dir = f'{outdir}/gen_edge_per_patch'
                edge_n_patches_out_dir = f'{outdir}/gen_edge_n_patches'
                os.makedirs(per_patch_out_dir, exist_ok=True)
                os.makedirs(n_patches_out_dir, exist_ok=True)
                os.makedirs(nuc_per_patch_out_dir, exist_ok=True)
                os.makedirs(nuc_n_patches_out_dir, exist_ok=True)
                os.makedirs(edge_per_patch_out_dir, exist_ok=True)
                os.makedirs(edge_n_patches_out_dir, exist_ok=True)
                get_csv(n_radius, testset, tsv_file_path, image_file_path, nuclei_file_path, edge_file_path,
                        per_patch_out_dir, n_patches_out_dir, nuc_per_patch_out_dir, nuc_n_patches_out_dir,
                        edge_per_patch_out_dir, edge_n_patches_out_dir)
        else:
            for j in range(3):
                testset = f'data/her2st/ST-imgs/{i}/{i + str(j+1)}'
                outdir = 'data/her2st'
                fig_name = os.listdir(testset)
                for h in range(3):
                    if fig_name[h].endswith("cat_image.jpg" or "seg.jpg"):
                        continue
                    else:
                        figname = fig_name[h]
                        break
                image_file_path = testset + '/' + figname
                tsv_file_path = f'{outdir}/ST-spotfiles/{i + str(j+1)}_selection.tsv'
                nuclei_file_path = f'{testset}/seg.jpg'
                edge_file_path = f'{testset}/cat_image.jpg'
                per_patch_out_dir = f'{outdir}/gen_per_patch'
                n_patches_out_dir = f'{outdir}/gen_n_patches'
                nuc_per_patch_out_dir = f'{outdir}/gen_nuc_per_patch'
                nuc_n_patches_out_dir = f'{outdir}/gen_nuc_n_patches'
                edge_per_patch_out_dir = f'{outdir}/gen_edge_per_patch'
                edge_n_patches_out_dir = f'{outdir}/gen_edge_n_patches'
                os.makedirs(per_patch_out_dir, exist_ok=True)
                os.makedirs(n_patches_out_dir, exist_ok=True)
                os.makedirs(nuc_per_patch_out_dir, exist_ok=True)
                os.makedirs(nuc_n_patches_out_dir, exist_ok=True)
                os.makedirs(edge_per_patch_out_dir, exist_ok=True)
                os.makedirs(edge_n_patches_out_dir, exist_ok=True)
                get_csv(n_radius, testset, tsv_file_path, image_file_path, nuclei_file_path, edge_file_path,
                        per_patch_out_dir, n_patches_out_dir, nuc_per_patch_out_dir, nuc_n_patches_out_dir,
                        edge_per_patch_out_dir, edge_n_patches_out_dir)
elif choice == 2:
    testset = 'data/stnet'
    for filename in os.listdir(testset + '/ST-imgs'):
        # 检查文件是否以.tif结尾
        if filename.endswith('.tif'):
            # 移除.tif后缀并打印文件名
            file = filename[:-4]
        tsv_file_path = f'{testset}/ST-spotfiles/{file[3:]}_selection.tsv'
        image_file_path = f'{testset}/ST-imgs/{file}.tif'
        nuclei_file_path = f'{testset}/ST-nuclei/{file}.jpg'
        edge_file_path = f'{testset}/ST-edge/{file}.jpg'
        per_patch_out_dir = f'{testset}/gen_per_patch'
        n_patches_out_dir = f'{testset}/gen_n_patches'
        nuc_per_patch_out_dir = f'{testset}/gen_nuc_per_patch'
        nuc_n_patches_out_dir = f'{testset}/gen_nuc_n_patches'
        edge_per_patch_out_dir = f'{testset}/gen_edge_per_patch'
        edge_n_patches_out_dir = f'{testset}/gen_edge_n_patches'
        os.makedirs(per_patch_out_dir, exist_ok=True)
        os.makedirs(n_patches_out_dir, exist_ok=True)
        os.makedirs(nuc_per_patch_out_dir, exist_ok=True)
        os.makedirs(nuc_n_patches_out_dir, exist_ok=True)
        os.makedirs(edge_per_patch_out_dir, exist_ok=True)
        os.makedirs(edge_n_patches_out_dir, exist_ok=True)

elif choice == 3:
    testset = 'data/skin'
    for filename in os.listdir(testset + '/ST-imgs'):
        # 检查文件是否以.tif结尾
        if filename.endswith('.jpg'):
            # 移除.tif后缀并打印文件名
            file = filename[:-4]
        tsv_file_path = f'{testset}/ST-spotfiles/{file[11:]}_selection.tsv'
        image_file_path = f'{testset}/ST-imgs/{file}.jpg'
        nuclei_file_path = f'{testset}/ST-nuclei/{file}.jpg'
        edge_file_path = f'{testset}/ST-edge/{file}.jpg'
        per_patch_out_dir = f'{testset}/gen_per_patch'
        n_patches_out_dir = f'{testset}/gen_n_patches'
        nuc_per_patch_out_dir = f'{testset}/gen_nuc_per_patch'
        nuc_n_patches_out_dir = f'{testset}/gen_nuc_n_patches'
        edge_per_patch_out_dir = f'{testset}/gen_edge_per_patch'
        edge_n_patches_out_dir = f'{testset}/gen_edge_n_patches'
        os.makedirs(per_patch_out_dir, exist_ok=True)
        os.makedirs(n_patches_out_dir, exist_ok=True)
        os.makedirs(nuc_per_patch_out_dir, exist_ok=True)
        os.makedirs(nuc_n_patches_out_dir, exist_ok=True)
        os.makedirs(edge_per_patch_out_dir, exist_ok=True)
        os.makedirs(edge_n_patches_out_dir, exist_ok=True)
else:
    dataset_choice = input("1.10x_breast_ff1\n2.10x_breast_ff2\n3.10x_breast_ff3")
    testset = 'data/test/10x_breast_ff' + str(dataset_choice)
    tsv_file_path = f'{testset}/ST-spotfiles/{testset}_selection.tsv'
    image_file_path = f'{testset}/ST-imgs/{testset}.tif'
    nuclei_file_path = f'{testset}/ST-nuclei/{testset}.jpg'
    edge_file_path = f'{testset}/ST-edge/{testset}.tif'
    per_patch_out_dir = f'{testset}/gen_per_patch'
    n_patches_out_dir = f'{testset}/gen_n_patches'
    nuc_per_patch_out_dir = f'{testset}/gen_nuc_per_patch'
    nuc_n_patches_out_dir = f'{testset}/gen_nuc_n_patches'
    edge_per_patch_out_dir = f'{testset}/gen_edge_per_patch'
    edge_n_patches_out_dir = f'{testset}/gen_edge_n_patches'
    os.makedirs(per_patch_out_dir, exist_ok=True)
    os.makedirs(n_patches_out_dir, exist_ok=True)
    os.makedirs(nuc_per_patch_out_dir, exist_ok=True)
    os.makedirs(nuc_n_patches_out_dir, exist_ok=True)
    os.makedirs(edge_per_patch_out_dir, exist_ok=True)
    os.makedirs(edge_n_patches_out_dir, exist_ok=True)


def get_csv(n_radius=0, testset, tsv_file_path, image_file_path, nuclei_file_path, edge_file_path, per_patch_out_dir, n_patches_out_dir, nuc_per_patch_out_dir, nuc_n_patches_out_dir,
            edge_per_patch_out_dir, edge_n_patches_out_dir):
    per_patch_file_path_list = []
    n_patches_file_path_list = []
    nuc_per_patch_file_path_list = []
    nuc_n_patches_file_path_list = []
    edge_per_patch_file_path_list = []
    edge_n_patches_file_path_list = []

    image = pyvips.Image.new_from_file(image_file_path)
    nuc = pyvips.Image.new_from_file(nuclei_file_path)
    edge = pyvips.Image.new_from_file(edge_file_path)

    with open(tsv_file_path, 'r') as tsv_file:
        lines = tsv_file.readlines()
        for line in lines[1:]:
            fields = line.strip().split('\t')

            print(fields)
            x = int(fields[3])
            y = int(fields[4])
            pixel_x = int(fields[5])
            pixel_y = int(fields[6])

            # 打开图像文件
            print('Image Width: ', image.width, ' Height: ', image.height)
            print('Nuc Width: ', nuc.width, ' Height: ', nuc.height)
            print('Edge Width: ', edge.width, ' Height: ', edge.height)

            # 生成单个patch，保存到本地
            left = pixel_x - radius_per_patch // 2
            top = pixel_y - radius_per_patch // 2
            print(f'left={left}, top={top}, x={x}, y={y}, radius={radius_per_patch}')
            patch = image.crop(top, left, radius_per_patch, radius_per_patch)
            per_patch_file_path = f'{per_patch_out_dir}/{testset}_{x}_{y}.tif'
            per_patch_file_path_list.append([x, y, pixel_x, pixel_y, per_patch_file_path])
            patch.write_to_file(per_patch_file_path)
            nuc_patch = nuc.crop(top, left, radius_per_patch, radius_per_patch)
            nuc_per_patch_file_path = f'{nuc_per_patch_out_dir}/{testset}_{x}_{y}.jpg'
            nuc_per_patch_file_path_list.append([x, y, pixel_x, pixel_y, nuc_per_patch_file_path])
            nuc_patch.write_to_file(nuc_per_patch_file_path)
            edge_patch = edge.crop(top, left, radius_per_patch, radius_per_patch)
            edge_per_patch_file_path = f'{edge_per_patch_out_dir}/{testset}_{x}_{y}.tif'
            edge_per_patch_file_path_list.append([x, y, pixel_x, pixel_y, edge_per_patch_file_path])
            edge_patch.write_to_file(edge_per_patch_file_path)

            # 生成邻域patch，保存到本地
            left = pixel_x - radius_per_patch * num_neighbors // 2
            top = pixel_y - radius_per_patch * num_neighbors // 2
            print(f'neighbor_left={left}, neighbor_top={top}, x={x}, y={y}, radius={n_radius}')
            n_patches = image.crop(top, left, n_radius, n_radius)
            n_patches_file_path = f'{n_patches_out_dir}/{testset}_n_{x}_{y}.tif'
            n_patches_file_path_list.append([x, y, pixel_x, pixel_y, n_patches_file_path])
            n_patches.write_to_file(n_patches_file_path)
            nuc_n_patches = nuc.crop(top, left, n_radius, n_radius)
            nuc_patches_file_path = f'{nuc_n_patches_out_dir}/{testset}_n_{x}_{y}.jpg'
            nuc_n_patches_file_path_list.append([x, y, pixel_x, pixel_y, nuc_patches_file_path])
            nuc_n_patches.write_to_file(nuc_patches_file_path)
            edge_n_patches = edge.crop(top, left, n_radius, n_radius)
            edge_patches_file_path = f'{edge_n_patches_out_dir}/{testset}_n_{x}_{y}.tif'
            edge_patches_file_path_list.append([x, y, pixel_x, pixel_y, edge_n_patches_file_path])
            edge_patches.write_to_file(edge_patches_file_path)

    print(len(per_patch_file_path_list))
    print(len(nuc_per_patch_file_path_list))
    print(len(n_patches_file_path_list))
    print(len(nuc_n_patches_file_path_list))
    print(len(edge_per_patch_file_path_list))
    print(len(edge_n_patches_file_path_list))

    # 保存单个patch信息到csv文件
    csv_file_path = f'{testset}/per_patch.csv'
    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['x', 'y', 'pixel-x', 'pixel-y', 'path'])
        for row in per_patch_file_path_list:
            writer.writerow(row)
    print(f'单个patch的数据已保存到 {csv_file_path}')

    # 保存邻域patch信息到csv文件
    csv_file_path = f'{testset}/n_patches.csv'
    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['x', 'y', 'pixel-x', 'pixel-y', 'path'])
        for row in n_patches_file_path_list:
            writer.writerow(row)
    print(f'邻域patch的数据已保存到 {csv_file_path}')

    csv_file_path = f'{testset}/nuc_per_patch.csv'
    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['x', 'y', 'pixel-x', 'pixel-y', 'path'])
        for row in nuc_per_patch_file_path_list:
            writer.writerow(row)
    print(f'单个nuc_patch的数据已保存到 {csv_file_path}')

    # 保存邻域patch信息到csv文件
    csv_file_path = f'{testset}/nuc_n_patches.csv'
    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['x', 'y', 'pixel-x', 'pixel-y', 'path'])
        for row in nuc_n_patches_file_path_list:
            writer.writerow(row)
    print(f'邻域nuc_patch的数据已保存到 {csv_file_path}')

    csv_file_path = f'{testset}/edge_per_patch.csv'
    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['x', 'y', 'pixel-x', 'pixel-y', 'path'])
        for row in edge_per_patch_file_path_list:
            writer.writerow(row)
    print(f'单个edge_patch的数据已保存到 {csv_file_path}')

    csv_file_path = f'{testset}/edge_n_patches.csv'
    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['x', 'y', 'pixel-x', 'pixel-y', 'path'])
        for row in edge_n_patches_file_path_list:
            writer.writerow(row)
    print(f'邻域edge_patch的数据已保存到 {csv_file_path}')
    return 0