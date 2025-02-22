import os
import zipfile
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

# Configurações iniciais
caminho_zip = 'KITTI-Sequence.zip'  
caminho_extracao = './KITTI_Sequence'  
caminho_ground_truth = 'ground_truth.npy'  

# Função para extrair o arquivo ZIP
def extrairZip(caminho_zip, caminho_extracao):
    if not os.path.exists(caminho_extracao):
        os.makedirs(caminho_extracao)
    with zipfile.ZipFile(caminho_zip, 'r') as zip_ref:
        zip_ref.extractall(caminho_extracao)

# Função para carregar a nuvem de pontos
def carregarNuvens(caminho):
    mesh = trimesh.load(caminho, process=False)
    return np.array(mesh.vertices)

# Normalização e pré-processamento da nuvem de pontos
def normalizarNuvem(nuvem):
    centroide = np.mean(nuvem, axis=0)
    return (nuvem - centroide) / np.std(nuvem, axis=0)

# Melhor ajuste de transformação entre duas nuvens
def ajustarTransformacao(A, B):
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    H = np.matmul(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.matmul(Vt.T, U.T)

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.matmul(Vt.T, U.T)

    t = centroid_B - np.matmul(R, centroid_A)

    return R, t

# Correspondência ponto a ponto entre nuvens
def correspondenciaPontos(src, dst, weights=False):
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    dist, indices = neigh.kneighbors(src)

    src_match = src
    dst_match = dst[indices.ravel()]

    if weights:
        weights = 1 / (1 + dist.ravel())
        return src_match, dst_match, dist.ravel(), weights

    return src_match, dst_match, dist.ravel()

# Cálculo do erro médio quadrático
def rmse(distances):
    return np.sqrt(np.mean(distances**2))

# Implementação do algoritmo ICP
def icp(A, B, max_iterations=50, min_error=1e-5, weights=False):
    src = np.copy(A)
    for i in range(max_iterations):
        if weights:
            src_match, dst_match, distances, w = correspondenciaPontos(src, B, weights=True)
            R, t = ajustarTransformacao(src_match * w[:, None], dst_match * w[:, None])
        else:
            src_match, dst_match, distances = correspondenciaPontos(src, B)
            R, t = ajustarTransformacao(src_match, dst_match)

        src = np.dot(src, R) + t

        error = rmse(distances)
        if error < min_error:
            break

    R, t = ajustarTransformacao(A, src)
    return R, t, error

# Função para estimar a trajetória
def estimarTrajetoria(caminho_pasta, max_iter=50, min_error=1e-5, weights=False):
    arquivos = []
    for root, _, files in os.walk(caminho_pasta):
        for file in files:
            if file.endswith('.obj'):
                arquivos.append(os.path.join(root, file))

    arquivos = sorted(arquivos)

    if not arquivos:
        raise FileNotFoundError("Nenhum arquivo .obj encontrado na pasta de entrada.")

    trajetoria = [np.eye(4)]
    nuvem_anterior = normalizarNuvem(carregarNuvens(arquivos[0]))

    for i in range(1, len(arquivos)):
        print(f"Processando ICP entre scans {i - 1} e {i}...")
        nuvem_atual = normalizarNuvem(carregarNuvens(arquivos[i]))
        R, t, error = icp(nuvem_anterior, nuvem_atual, max_iter, min_error, weights)
        print(f"Erro médio entre scans {i - 1} e {i}: {error:.6f}")

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        trajetoria.append(trajetoria[-1] @ T)

        nuvem_anterior = nuvem_atual

    return np.array(trajetoria)

# Função para visualizar a trajetória
def plotarTrajetoria(trajetoria, ground_truth):
    trajetoria_xyz = trajetoria[:, :3, 3]
    ground_truth_xyz = ground_truth[:, :3, 3]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(trajetoria_xyz[:, 0], trajetoria_xyz[:, 1], trajetoria_xyz[:, 2], label='Estimada', color='blue')
    ax.plot(ground_truth_xyz[:, 0], ground_truth_xyz[:, 1], ground_truth_xyz[:, 2], label='Ground Truth', color='red')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    plt.title("Trajetória Estimada vs Ground Truth")
    plt.show()

# Pipeline principal
if __name__ == '__main__':
    dispositivo = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Extração do arquivo ZIP
    print("Extraindo arquivo ZIP...")
    extrairZip(caminho_zip, caminho_extracao)

    # Carregar ground truth
    ground_truth = np.load(caminho_ground_truth)

    # Estimar trajetória
    print("Estimando trajetória...")
    trajetoria = estimarTrajetoria(caminho_extracao, weights=True)

    # Visualizar resultados
    plotarTrajetoria(trajetoria, ground_truth)

    # Exibir última matriz de transformação
    print("Última matriz de transformação estimada:")
    print(trajetoria[-1])

    # Exibir matriz ground truth correspondente
    print("Matriz de transformação ground truth correspondente:")
    print(ground_truth[-1])