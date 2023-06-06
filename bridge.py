import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, OPTICS
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.neighbors import NearestNeighbors
from sklearn import linear_model
import os
from scipy.spatial import ConvexHull, Delaunay
import warnings

warnings.simplefilter('ignore', np.RankWarning)

def loadData(idx: int):
    return [pd.read_csv('Bridge/bridge_lidar-points/' + file).to_numpy()[:, :3] for file in os.listdir('Bridge/bridge_lidar-points')][idx]

def xyz(X: np.ndarray):
    return X[:, 0], X[:, 1], X[:, 2]

def show(X: np.ndarray, labels = None):
    fig, ax = plt.subplots(subplot_kw = {"projection": "3d"})
    ax.scatter(*xyz(X), c = labels, marker = '.')
    plt.show()

def scale(X: np.ndarray, plot = False):
    scaled = RobustScaler().fit_transform(X)
    # scaled = StandardScaler().fit_transform(X)
    if plot:
        show(scaled)
    
    return scaled

def iscale(X: np.ndarray, plot = False):
    inverse = RobustScaler().fit(X).inverse_transform(X)

    if plot:
        show(inverse)

    return inverse

def cluster(X, k = 3, plot = True):

    k_means = KMeans(init = "k-means++", n_clusters = k, n_init = 10)
    k_means.fit(X)

    k_means_cluster_centers = k_means.cluster_centers_
    k_means_labels = pairwise_distances_argmin(X, k_means_cluster_centers)

    if plot:
        show(X, k_means_labels)

    return k_means_labels

def GM(X, k = 3, plot = True):

    gmm = GaussianMixture(n_components = k, random_state = 0)
    gmm.fit(X)
    labels = gmm.predict(X)

    if plot:
        show(X, labels)
        # fig, ax = plt.subplots(subplot_kw = {"projection": "3d"})
        # ax.scatter(*xyz(X), c = labels, marker = '.')
        # plt.show()

    return labels

def DB(X, eps = 0.5, min_samples = 5, plot = True) -> np.ndarray:
    dbscan = DBSCAN(eps = eps, min_samples = min_samples).fit(X)
    # f = dbscan.fit(X)
    # labels = dbscan.fit_predict(X)
    labels = dbscan.labels_

    if plot:
        show(X, labels)
        # fig, ax = plt.subplots(subplot_kw = {"projection": "3d"})
        # ax.scatter(*xyz(X), c = labels, marker = '.')
        # plt.show()

    return labels

def Optcs(X, eps = np.inf, min_samples = 5, plot = True):
    op = OPTICS(min_samples = min_samples, max_eps = eps, cluster_method= 'dbscan').fit(X)
    labels = op.labels_

    if plot:
        show(X, labels)

    return labels

def WCSS(X, k = 6, plot = True):
    km_scores = np.zeros(k)
    for i in range(1, k + 1):
        k_means = KMeans(init="k-means++", n_clusters = i, n_init=10)
        k_means.fit(X)
        km_scores[i - 1] = k_means.inertia_

    if plot:
        plt.plot([i + 1 for i in range(k)], km_scores, 'o-')
        plt.xlabel('Number of clusters')
        plt.ylabel('Score')
        plt.title('WCSS Curve')
        plt.show()

    return km_scores

def elbow(X, k = 6, plot = True):
    km_scores = np.zeros(k)
    for i in range(1, k + 1):
        k_means = KMeans(init="k-means++", n_clusters = i, n_init=10)
        k_means.fit(X)
        km_scores[i - 1] = -k_means.score(X)

    if plot:
        # plt.subplot(1, 2, 1)
        plt.plot([i + 1 for i in range(k)], km_scores, 'o-')
        plt.xlabel('Number of clusters')
        plt.ylabel('Score')
        plt.title('Elbow Curve')
        
        # plt.subplot(1, 2, 2)
        # plt.plot([i + 1 for i in range(k - 1)], deriv([i for i in range(k)], km_scores), 'o-')
        # # plt.xlabel('Number of clusters')
        # # plt.ylabel('Score')
        # # plt.title('Elbow Curve')
        
        plt.show()

    return km_scores

def silhouette(X, k = 6, plot = True):
    km_scores = np.zeros(k)
    for i in range(2, k + 2):
        k_means = KMeans(init="k-means++", n_clusters = i, n_init=10)
        k_means.fit(X)
        k_means_cluster_centers = k_means.cluster_centers_
        k_means_labels = pairwise_distances_argmin(X, k_means_cluster_centers)
        # print(np.all(k_means_labels == k_means.predict(X)))
        km_scores[i - 2] = silhouette_score(X, k_means_labels)

    if plot:
        plt.plot([i + 2 for i in range(k)], km_scores, 'o-')
        plt.xlabel('Number of clusters')
        plt.ylabel('Score')
        plt.title('Silhouette Curve')
        plt.show()
    
    return km_scores

def DBCV(X, labels, n = 10):
    # Calculate the intra-cluster distances
    intra_cluster_distances = []
    for label in np.unique(labels):
        cluster_points = X[labels == label]
        if len(cluster_points) > 1:
            nn = NearestNeighbors(n_neighbors = n).fit(cluster_points)
            distances, _ = nn.kneighbors(cluster_points)
            mean_distance = np.mean(distances[:, -1])
            intra_cluster_distances.append(mean_distance)

    # Calculate the inter-cluster distances
    inter_cluster_distances = []
    nn = NearestNeighbors(n_neighbors = n).fit(X)
    distances, indices = nn.kneighbors(X)
    for i in range(len(X)):
        label = labels[i]
        other_labels = labels[indices[i]]
        different_cluster_distances = distances[i][other_labels != label]
        if len(different_cluster_distances) > 0:
            min_distance = np.min(different_cluster_distances)
            inter_cluster_distances.append(min_distance)

    # Calculate the DBCV index
    intra_mean = np.mean(intra_cluster_distances) if intra_cluster_distances else 0.0
    inter_mean = np.mean(inter_cluster_distances) if inter_cluster_distances else np.inf
    dbcv = inter_mean / intra_mean if intra_mean != 0.0 else np.inf
    return dbcv

def selectDBCV(X, kmax = 5, n = 10, plot = False):
    scores = np.zeros(kmax)
    best_k = 1
    best_score = np.inf

    for k in range(2, kmax + 1):
        gmm = GaussianMixture(n_components = k, random_state = 0)
        gmm.fit(X)
        score = DBCV(X, gmm.predict(X), n)
        scores[k - 1] = score

        if score < best_score:
            best_score = score
            best_k = k

    if plot:
        plt.plot([i + 1 for i in range(kmax)], scores, 'o-')
        plt.xlabel('Number of clusters')
        plt.ylabel('Score')
        plt.title('DBCV Curve')
        plt.show()

    return best_k

def selectGM(X, kmax = 5, metric = 'bic', threshold = .8, plot = True):
    scores = np.zeros(kmax)
    best_k = 1

    gmm = GaussianMixture(n_components = 1, random_state = 0)
    gmm.fit(X)
    if metric == 'bic':
        best_score = gmm.bic(X)
    
        scores[0] = best_score

        for k in range(2, kmax):
            gmm = GaussianMixture(n_components = k, random_state = 0)
            gmm.fit(X)
            score = gmm.bic(X)
            scores[k - 1] = score

            if score < best_score:
                best_score = score
                best_k = k
    elif metric == 'silhouette':
        for k in range(2, kmax):
            gmm = GaussianMixture(n_components = k, random_state = 0)
            gmm.fit(X)
            scores[k - 1] = silhouette_score(X, gmm.predict(X))
        print(scores)
        if max(scores) > threshold:
            best_k = np.argmax(scores) + 2
    elif metric == 'aic':
        best_score = gmm.bic(X)
        
        scores[0] = best_score

        for k in range(2, kmax):
            gmm = GaussianMixture(n_components = k, random_state = 0)
            gmm.fit(X)
            score = gmm.aic(X)
            scores[k - 1] = score

            if score < best_score:
                best_score = score
                best_k = k

    if plot:
        pass
        # plt.plot([i + 1 for i in range(kmax)], scores, 'o-')
        # plt.xlabel('Number of clusters')
        # plt.ylabel('Score')
        # plt.title('BIC Curve')
        # plt.show()

    return best_k

def selectK(X, kmax = 5, threshold = .8):
    scores = silhouette(X, kmax, False)
    if max(scores) > threshold:
        return np.argmax(scores) + 2
    else:
        return 1

def separate(X: np.ndarray, labels: np.ndarray, plot = False, threshold = 2e2):
    k = len(np.unique(labels[labels != -1]))
    Xsep = [X[labels == i] for i in range(k) if len(np.unique(X[labels == i])) > threshold]
    # print([len(np.unique(d)) for d in Xsep])

    if plot:
        fig, ax = plt.subplots(1, k, figsize = (20, 5))
        for i in range(k):
            ax[i].scatter(Xsep[i][:, 0], Xsep[i][:, 1])
        plt.show()

    return np.asarray(Xsep, dtype = object)

def isBridge(bridges: np.ndarray, threshold = 15) -> np.ndarray:
    indices = np.where([np.mean([np.var(bridge[:, 0]), np.var(bridge[:, 1])]) > threshold for bridge in bridges])[0]
    # print(indices)
    return indices.tolist()

def IQR(bridges: np.ndarray) -> np.ndarray:
    '''
    Returns the IQR and 1st, 3rd quadrants of the z values of each bridge
    '''
    stats = np.zeros((len(bridges), 3))
    for idx, bridge in enumerate(bridges):
        stats[idx, 0] = np.percentile(bridge[:, 2], 25) # Q1
        stats[idx, 1] = np.percentile(bridge[:, 2], 75) # Q3
        stats[idx, 2] = stats[idx, 1] - stats[idx, 0] # IQR

    return stats

def bridgePoints(bridges: np.ndarray, threshold = 15, onlyBridges = True, plot = False) -> np.ndarray:
    if plot:
        unfiltered = np.copy(bridges)
    # print(np.shape(bridges))
    if onlyBridges:
        bridges = bridges[isBridge(bridges, threshold)]

    stats = IQR(bridges)

    for idx, bridge in enumerate(bridges):
        non_outliers = np.logical_and(bridge[:, 2] > stats[idx, 0] - 1.5 * stats[idx, 2], bridge[:, 2] < stats[idx, 1] + 1.5 * stats[idx, 2])
        bridge = bridge[non_outliers]

    if plot:
        fig, ax = plt.subplots(subplot_kw = {"projection": "3d"})
        for raw in unfiltered:
            ax.scatter(*xyz(raw), marker = '.', color = 'r')
            # ax.scatter(raw[:,0], raw[:,1], raw[:,2], marker = '.', color = 'r')
        for bridge in bridges:
            ax.scatter(*xyz(bridge), marker = '.')
            # ax.scatter(bridge[:,0], bridge[:,1], bridge[:,2], marker = '.')
        plt.show()

    return bridges

def regression(bridge: np.ndarray, plot = True):
    x, y, z = xyz(bridge)
    X = x.reshape(-1, 1)
    Y = y.reshape(-1, 1)
    fit = linear_model.LinearRegression().fit(X, Y)
    
    if plot:
        plt.scatter(x, y, c = z)
        plt.plot(x, fit.predict(x.reshape(-1, 1)), color = 'r')
        plt.show()

    return fit.coef_, fit.score(X, Y)
    
def cnvxHull(bridge: np.ndarray, plot = True):
    hull = ConvexHull(bridge[:, :2])

    # Access the vertices of the convex hull
    indices = hull.vertices
    
    if plot:
        x, y, z = xyz(bridge)
        plt.scatter(x, y, c = z)
        plt.scatter(x[indices], y[indices], c = 'r', alpha = 0.5)
        plt.show()
    
    return indices

def alpha_shape(points, alpha = 0.1, plot = True):
    # Compute Delaunay triangulation
    tri = Delaunay(points)

    # Compute circumcircle radii of triangles
    circum_radii = []
    for simplex in tri.simplices:
        vertices = points[simplex]
        center = np.mean(vertices, axis=0)
        radius = np.max(np.linalg.norm(vertices - center, axis=1))
        circum_radii.append(radius)
    circum_radii = np.array(circum_radii)

    # Select triangles below the alpha threshold
    selected_tri = tri.simplices[circum_radii < alpha]

    # Flatten and unique the indices of selected triangles
    indices = np.unique(selected_tri.flatten())

    # Compute convex hull of selected points
    hull = ConvexHull(points[indices])
    
    if plot:
        plt.scatter(points[:, 0], points[:, 1], color='blue', alpha=0.5)
        plt.plot(points[hull.vertices, 0], points[hull.vertices, 1], color='red', lw=2)
        plt.axis('equal')
        plt.show()
    return hull

def extremes(bridge: np.ndarray, plot = False):
    '''
    Returns : indices of corners of bridge
    '''
    x, y, z = xyz(bridge)

    xmin, xmax, ymin, ymax = np.argmin(x), np.argmax(x), np.argmin(y), np.argmax(y)
    
    indices = np.array([xmin, xmax, ymin, ymax])

    if plot:
        plt.scatter(x, y, c = z, marker = '.')
        plt.scatter(x[indices], y[indices], c = 'r', marker = '.')
        plt.show()

    return xmin, xmax, ymin, ymax

def line(p1, p2, d = 1):
    return np.polyfit([p1[0], p2[0]], [p1[1], p2[1]], deg = d)

def dist(p1, p2):
    return abs(np.linalg.norm(p1[:2] - p2[:2]))

def containment(bridge: np.ndarray, corners: np.ndarray, d = 1, plot = True, idx = None, bNum = None, total = None):
    xmin, xmax, ymin, ymax = corners
    
    x, y, z = xyz(bridge)
    
    r, b, l, t = (bridge[ymax], bridge[xmax]), (bridge[xmax], bridge[ymin]), (bridge[ymin], bridge[xmin]), (bridge[xmin], bridge[ymax])
    
    id1 = np.argmin([dist(*r), dist(*b), dist(*l), dist(*t)])
    id2 = (id1 + 2) % 4
    
    # picks the shortest side and the oppisite side as the entrance and exit for the bridge (MIGHT NOT BE VALID FOR SOME BRIDGES)
    entrance = np.array([r, b, l, t])[[id1, id2]] # shape (2, 2, 3)

    right   = np.polyval(line(*r, d), x)
    bottom  = np.polyval(line(*b, d), x)
    left    = np.polyval(line(*l, d), x)
    top     = np.polyval(line(*t, d), x)

    mask = np.logical_and(
        np.logical_and(y < right, 
                       y > bottom), 
        np.logical_and(y > left, 
                       y < top))

    if plot:
        s = .01
        m = '.'
        color = 'k'
        
        rMask = (x >= r[0][0]) & (x <= r[1][0])
        bMask = (x <= b[0][0]) & (x >= b[1][0])
        lMask = (x <= l[0][0]) & (x >= l[1][0])
        tMask = (x >= t[0][0]) & (x <= t[1][0])

        plt.scatter(x, y, c = z, marker = m)
        # plt.scatter([r[0][0], b[0][0], l[0][0], t[0][0]], [r[0][1], b[0][1], l[0][1], t[0][1]], c = 'r', marker = m)
        plt.scatter(x[mask], y[mask], c = 'r', marker = m)
        plt.plot(x[rMask], right[rMask], c = color)
        plt.plot(x[bMask], bottom[bMask], c = color)
        plt.plot(x[lMask], left[lMask], c = color)
        plt.plot(x[tMask], top[tMask], c = color)
        plt.xlim(x[xmin] - s, x[xmax] + s)
        plt.ylim(y[ymin] - s, y[ymax] + s)
        plt.axis('off')
        plt.title(f'Dataset {idx + 1} \n Bridge {bNum + 1}/{total}')
        plt.show()

    contained = bridge[mask]
    area = len(contained) / len(bridge) # proportion of points contained in hull

    return contained, area, entrance

def rotate(bridge: np.ndarray, angle: float, center = None, plot = False):
    rad = np.radians(angle)
    x, y, z = xyz(bridge)
    
    if center:
        X, Y = center
    else:
        X, Y, _ = np.mean(bridge, axis = 0)
    xSub, ySub = x - X, y - Y
    xR, yR = X + xSub * np.cos(rad) - ySub * np.sin(rad), Y + xSub * np.sin(rad) + ySub * np.cos(rad)
    bridgeRot = np.vstack((xR, yR, z)).T
    
    if plot:
        m = '.'
        plt.scatter(x, y, c = z, marker = m)
        plt.scatter(xR, yR, c = z, marker = m)
        plt.show()

    return bridgeRot, X, Y

def findEdges(bridge: np.ndarray, idx: int = None, bNum = None, total = None, threshold = .7, angle = 90, iterations = 10, plot = False):
    vertices = extremes(bridge)
    _, area, entrance = containment(bridge, vertices, plot = False)
    
    
    if area < threshold:
        angle = np.linspace(start = angle / iterations, stop = angle, num = iterations)
        areas = np.zeros(iterations)
        corners = np.zeros((iterations, 4), dtype = int)
        entrances = np.zeros((iterations, 2, 2, 3))

        for i in range(iterations):
            bridgeRot, X, Y = rotate(bridge, angle[i], plot = False)
            corners[i] = extremes(bridgeRot, plot = False)
            _, areas[i], entrances[i] = containment(bridgeRot, corners[i], plot = False, idx = idx, bNum = bNum, total = total)

        best = np.argmax(areas)

        bridge, X, Y = rotate(bridge, angle[best], plot = False)
        
        vertices = corners[best]
        # entrance = entrances[best]
        entrance = [rotate(entrances[best][i], -angle[best], center = (X, Y))[0] for i in range(2)]
        
        # print(area, areas)
        area = max(areas[best], area)
    
    if area < threshold:
        # containment(bridge, vertices, d = 2, plot = True)
        print(f'WARNING (Dataset {idx + 1}): area is still less than threshold!')
    # print(area)

    if plot:
        containment(bridge, vertices, plot = True, idx = idx, bNum = bNum, total = total)

    return vertices, entrance

def distance(P1, P2, Q):
    '''
    Distance from point to line (2D)
    '''
    
    p1, p2, q = P1[:2], P2[:2], Q[:2]

    return abs(np.linalg.norm(np.cross(p2 - p1, p1 - q)) / np.linalg.norm(p2 - p1))

def Z(bridge: np.ndarray, edges: np.ndarray, prop = .05, idx = None, bNum = None, total = None, plot = False):
    e1, e2 = edges

    eP1 = np.argsort([distance(*e1, p) for p in bridge])
    eP2 = np.argsort([distance(*e2, p) for p in bridge])

    n = round(prop * len(bridge))

    mask = np.concatenate((eP1[:n], eP2[:n]))


    if plot:
        x, y, z = xyz(bridge)
        m = '.'

        plotMask = [i not in mask for i in range(len(bridge))]

        plt.scatter(x, y, c = z, marker = m)
        plt.scatter(x[mask], y[mask], c = 'r', marker = m)
        plt.title(f'Dataset {idx + 1} \n Bridge {bNum + 1}/{total}')
        plt.show()
    
        fig, ax = plt.subplots(subplot_kw = {"projection": "3d"})
        ax.scatter(x[plotMask], y[plotMask], z[plotMask], marker = '.')
        ax.scatter(x[mask], y[mask], z[mask], c = 'r', marker = m)
        plt.title(f'Dataset {idx + 1} \n Bridge {bNum + 1}/{total}')
        plt.show()

    

if __name__ == '__main__':
    
    # need to slice 15 into pieces?  Or just consider close to extremals somehow?

    # idx = 11
    # problematic 15
    
    for idx in range(19):
        X = scale(loadData(idx))
        labels = DB(X, .5, plot = False)
        bridges = separate(X, labels)
        tot = len(bridges)
        
        for i,bridge in enumerate(bridges):
            _, edges = findEdges(bridge, idx = idx, bNum = i, total = tot, plot = False)
            Z(bridge, edges, idx = idx, bNum = i, total = tot, plot = True)