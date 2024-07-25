import cv2
import math
import matplotlib.pyplot as plt

# p1, p2사이의 유클리드 거리 
def euclidean_distance(p1, p2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

# K-Means 알고리즘
def KMeans(image, k):
    rows = len(image)
    cols = len(image[0])
    dims = len(image[0][0]) #색상의 수 -> RGB는 3
    image2 = [pixel for row in image for pixel in row]  # 이미지를 1차원 리스트로 변환
    
    col2 = rows * cols #이미지의 총 픽셀 수
    
    # k개의 클러스터 중심 초기화
    rand_k = []
    #rand_k = [[random.randint(0, 255) for _ in range(dims)] for _ in range(k)] # 랜덤으로 초기화
    
    # k개의 각 클러스터의 중심이 일정한 최대 거리만큼 분포하도록 초기화
    # 클러스터 중심의 시작 점과 끝 점은 (0, 0, 0), (255, 255, 255)가 되도록 함
    interval = 256
    if k != 1:
        interval = 255 / (k - 1)
    i = 0
    while i < 256:
        ii = round(i)
        rand_k.append([ii, ii, ii])
        print(ii)
        i += interval

    cluster_info = [0] * col2
    
    # 각 픽셀을 가장 가까운 클러스터 중심에 할당 과정
    while True:
        #한 픽셀과 각 클러스터 중심 사이의 거리를 저장하는 리스트
        clust_distance = [[0] * k for _ in range(col2)] 
        
        #각 픽셀에 대해, 모든 클러스터 중심과의 거리를 계산하고, 가장 가까운 클러스터 중심의 '인덱스'를 cluster_info에 저장
        for i in range(col2):
            for h in range(k):
                clust_distance[i][h] = euclidean_distance(image2[i], rand_k[h])
        new_cluster_info = [min(range(k), key=lambda h: clust_distance[i][h]) for i in range(col2)]
        #각 픽셀에 대해 가장 가까운 클러스터 중심(인덱스)를 반환

        new_rand_k = [[0] * dims for _ in range(k)] # 새로운 클러스터 중심을 저장할 리스트
        counts = [0] * k    # 각 클러스터에 할당된 픽셀 수를 세는 리스트
        
        # 각 클러스터에 속한 픽셀들의 평균 값을 계산하여 새로운 클러스터 중심을 구하기
        for i in range(col2):
            cluster = new_cluster_info[i]
            for d in range(dims):
                new_rand_k[cluster][d] += image2[i][d] # 현재 픽셀의 색상 값을 해당 클러스터의 색상 값 합계에 누적
            counts[cluster] += 1
        
        for cluster in range(k):
            if counts[cluster] == 0:
                counts[cluster] = 1
            for d in range(dims):
                new_rand_k[cluster][d] /= counts[cluster]
        
        # 클러스터의 기존 중심과 새로운 중심의 거리가 10^5 미만이라면, 값 갱신 후 종료
        if all(euclidean_distance(rand_k[h], new_rand_k[h]) < 1e-5 for h in range(k)):
            rand_k = new_rand_k
            cluster_info = new_cluster_info
            break
        
        # 클러스터의 중심 값 갱신
        rand_k = new_rand_k
        cluster_info = new_cluster_info
    
    result_cluster = [[0] * dims for _ in range(col2)]
    
    #각 픽셀이 속한 클러스터의 색상 값을 저장
    for i in range(col2):
        result_cluster[i] = rand_k[cluster_info[i]] #클러스터 중앙값으로 다 변환
    
    #이차원으로 변환
    image3 = []
    for i in range(rows):
        row = []
        for j in range(cols):
            row.append(result_cluster[i * cols + j])
        image3.append(row)
    
    # 결과 이미지, 각 클러스터의 중심, 클러스터 정보를 반환
    return image3, rand_k, cluster_info

def convert_BGR_to_RGB(BGR):
    rows = len(BGR)
    cols = len(BGR[0])
    RGB = []
    for row in BGR:
        RGB_row = []
        for pixel in row:
            B, G, R = pixel
            # Convert BGR to RGB
            r = R
            g = G
            b = B
            RGB_row.append([r, g, b])
        RGB.append(RGB_row)
    return RGB


# 엘보우 방법을 위한 sse값 계산
def calculate_sse(image, rand_k, cluster_info):
    sse = 0
    for i, pixel in enumerate(image):
        cluster = cluster_info[i]
        sse += euclidean_distance(pixel, rand_k[cluster]) ** 2
    return sse

# 엘보우 방법
def elbow(image, rand_k, cluster_info):
    rows = len(image)
    cols = len(image[0])
    dims = len(image[0][0])
    image2 = [pixel for row in image for pixel in row] # 2차원 이미지에 대한 1차원 배열
    
    return calculate_sse(image2, rand_k, cluster_info)
    
# 실루엣 방법
def silhouette(image, k:int, clusterAvg, clusterInfo):
    rows = len(image)
    cols = len(image[0])
    dims = len(image[0][0]) #색상의 수 -> RGB는 3
    image2 = [pixel for row in image for pixel in row]  # 이미지를 1차원 리스트로 변환
    
    col2 = rows * cols # 이미지의 총 픽셀 수
    for i in range(col2):
        curDistance = -1
        minIndex = -1
        sumOfSameCluster = 0
        cntOfSameCluster = 0
        sumOfDifferCluster = 0
        cntOfDifferCluster = 0
        
        for j in range(k):
            if j == clusterInfo[i]:
                continue
            distance = euclidean_distance(image2[i], clusterAvg[j])
            if distance < curDistance or curDistance == -1:
                curDistance = distance
                minIndex = j
        for j in range(col2):
            if i == j:
                continue
            if clusterInfo[j] == clusterInfo[i]:
                sumOfSameCluster += euclidean_distance(image2[i], image2[j])
                cntOfSameCluster += 1
            elif clusterInfo[j] == minIndex:
                sumOfDifferCluster += euclidean_distance(image2[i], image2[j])
                cntOfDifferCluster += 1
        avgOfSameCluster = sumOfSameCluster / cntOfSameCluster
        avgOfDeferCluster = sumOfDifferCluster / cntOfDifferCluster

        silhouetteScore = (avgOfDeferCluster - avgOfSameCluster) / max(avgOfSameCluster, avgOfDeferCluster)
        return silhouetteScore


image = cv2.imread('C:/Users/test.jpg')
image = convert_BGR_to_RGB(image)  # BGR을 RGB로 변환

#k = 5
outImg = []
silhouetteScores = []
max_k = 10
sse = []
# outImg의 범위는 [0, 9)의 인덱스, k의 범위는 2 ~ 10 (각 0 ~ 8의 인덱스로 매핑)
for i in range(1, max_k + 1):
    out, clusterAvg, clusterInfo = KMeans(image, i)
    sseK = elbow(image, clusterAvg, clusterInfo)
    sse.append(sseK)
    print("i :", i)
    print(clusterAvg)
    if i == 1:
        continue
    outImg.append(out) #K=2 부터
    silhouetteScores.append(silhouette(image, i, clusterAvg, clusterInfo))

print("silhousetteScores :", silhouetteScores)
print("sse :", sse)

# Elbow Method의 best K 구하기
sse_diff = [abs(sse[i] - sse[i - 1]) for i in range(1, len(sse))]
sse_diff_va = [abs(sse_diff[i] - sse_diff[i - 1]) for i in range(1, len(sse_diff))]
elbow_best_k = sse_diff_va.index(max(sse_diff_va)) + 1
outputElbow = outImg[elbow_best_k]


# Silhousette의 best K 구하기
silhousetteBestK = silhouetteScores.index(max(silhouetteScores))
outputSilhousette = outImg[silhousetteBestK]

outputSilhousette = [[list(map(int, pixel)) for pixel in row] for row in outputSilhousette]  # Convert float to int
outputElbow = [[list(map(int, pixel)) for pixel in row] for row in outputElbow]  # Convert float to int

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes[0, 0].imshow(image)
axes[0, 0].set_title("Original Image")
axes[0, 0].axis('off')

axes[0, 2].imshow(outputElbow)
axes[0, 2].set_title("Elbow Method Best K = " + str(elbow_best_k + 2))
axes[0, 2].axis('off')

# 엘보우 메소드 그래프
axes[0, 1].plot(range(1, max_k + 1), sse, 'bo-')
axes[0, 1].set_title("Elbow Method")
axes[0, 1].set_xlabel("Number of Clusters")
axes[0, 1].set_ylabel("SSE")
axes[0, 1].set_xticks(range(1, max_k + 1))

# 실루엣 점수 그래프
axes[1, 1].set_title('Silhouette Method')
axes[1, 1].plot(range(2, max_k + 1), silhouetteScores, 'bo-')
axes[1, 1].set_xlabel("Number of Clusters")
axes[1, 1].set_ylabel("Silhouette Scores") 
axes[1, 1].set_xticks(range(2, max_k + 1))

axes[1, 2].imshow(outputSilhousette)
axes[1, 2].set_title("Silhouette Method Best K = " + str(silhousetteBestK + 2))

# 나머지 서브플롯 숨기기
for ax in [axes[1, 0], axes[1, 2]]:
    ax.axis('off')

plt.tight_layout()
plt.show()
