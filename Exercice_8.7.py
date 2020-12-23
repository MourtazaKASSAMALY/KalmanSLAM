from roblib import *


M = loadcsv("slam_data.csv")
t, phi, teta, psi, vr, pz, a = M[:, 0], M[:, 1], M[:, 2], M[:, 3], M[:, 4:7].T, M[:, 7], M[:, 8]
kmax = len(M)

dt = 0.1
xhat = np.zeros((15, 1))  # posx, posy, posz du robot et les coordonnées des 6 amers à estimer
p = np.zeros((3, 1))  # posx, posy, posz du robot pour la dynamique
Gamma = 10**6 * diag([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
A = np.eye(15, 15)
sigmav = 1
GammaAlphaX = (dt*sigmav)**2 * diag([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

xforward = [xhat]
Gammaforward = [Gamma]
xupdate = []
Gammaupdate = []
xback = [None] * kmax
Gammaback = [None] * kmax


# -------------------------------------------------------------------------------------------------------------------- #


def kalman(C, Gamma, GammaBeta, y, x, A, u, GammaAlphaX):
    # Cas où le filtre de Kalman s'utilise en mode prédiction uniquement
    if GammaBeta == [] and y == [] and C == []:
        GammaBeta = np.eye(3, 3)
        C = np.zeros((3, 15))
        y = np.eye(1, 1)

    # Phase de correction
    S = C @ Gamma @ C.T + GammaBeta
    K = Gamma @ C.T @ np.linalg.inv(S)
    ytilde = y - C @ x

    Gammanext = Gamma - K @ C @ Gamma
    xnext = x + K @ ytilde

    # Phase de prédiction
    xpredic = A @ xnext + u
    gammapredic = A @ Gammanext @ A.T + GammaAlphaX
    return xnext, Gammanext, xpredic, gammapredic


# -------------------------------------------------------------------------------------------------------------------- #


def obs(i, dt):
    T = [1054, 1092, 1374, 1748, 3038, 3688, 4024, 4817, 5172, 5232, 5279, 5688]  # instants de détection des amer
    N = [1, 2, 1, 0, 1, 5, 4, 3, 3, 4, 5, 1]  # numéros des amers
    r = [52.42, 12.47, 54.40, 52.68, 27.73, 26.98, 37.90, 36.71, 37.37, 31.03, 33.51, 15.05]  # distance des amers

    if i * dt in T:  # si un amer est détecté
        j = T.index(i * dt)
        k, r = N[j], r[j]
        yi = R @ array([[0], [-sqrt(r ** 2 - a[i] ** 2)], [-a[i]]])
        y = vstack((yi[0:2, :], pz[i]))
        Ci = hstack((eye(2, 3), zeros((2, 2*k)), -eye(2), zeros((2, 12-2*(k+1)))))
        C = vstack((Ci, np.array([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])))
        GammaBeta = 0.1 * eye(3, 3)
        return y, C, GammaBeta

    else:  # sinon
        C = np.array([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        y = np.array([[pz[i]]])
        GammaBeta = 0.1 * eye(1, 1)
        return y, C, GammaBeta


# -------------------------------------------------------------------------------------------------------------------- #


if __name__ == "__main__":

    for i in range(kmax):
        R = eulermat(phi[i], teta[i], psi[i])
        v = vr[:, i].reshape(3, 1)

        p = p + dt * R @ v  # dynamique

        y, C, GammaBeta = obs(i, dt)  # mesures/observations
        u = vstack((dt * R @ v, zeros((12, 1))))
        xup, Gammaup, xhat, Gamma = kalman(C, Gamma, GammaBeta, y, xhat, A, u, GammaAlphaX)  # kalman

        xupdate.append(xup)
        Gammaupdate.append(Gammaup)
        xforward.append(xhat)
        Gammaforward.append(Gamma)

    # Lissage des valeurs pour post-traitement et réductions des incertitudes

    xback[kmax-1] = xup
    Gammaback[kmax-1] = Gammaup
    for k in arange(kmax-2, -1, -1):
        J = Gammaupdate[k] @ A.T @ np.linalg.inv(Gammaforward[k+1])
        xback[k] = xupdate[k] + J @ (xback[k+1] - xforward[k+1])
        Gammaback[k] = Gammaupdate[k] + J @ (Gammaback[k+1] - Gammaforward[k+1]) @ J.T

    # Tracé des ellipsoides de confiance de la trajectoire lissée

    ax = init_figure(-100, 1000, -100, 1000)

    for k in range(kmax):
        if k % 100 == 0:
            draw_ellipse(xback[k][0:2], Gammaback[k][0:2, 0:2], 0.99, ax, "blue", "None")
            pause(0.01)

    # Tracé des amers et de leurs ellipsoides de confiance

    mx = [xback[kmax-1][3, 0], xback[kmax-1][5, 0], xback[kmax-1][7, 0],
          xback[kmax-1][9, 0], xback[kmax-1][11, 0], xback[kmax-1][13, 0]]
    my = [xback[kmax-1][4, 0], xback[kmax-1][6, 0], xback[kmax-1][8, 0],
          xback[kmax-1][10, 0], xback[kmax-1][12, 0], xback[kmax-1][14, 0]]
    for i in range(len(mx)):
        draw_ellipse((mx[i], my[i]), Gamma[2+i:2+i+2, 2+i:2+i+2], 0.99, ax, "red", "None")
        ax.plot(mx[i], my[i], 'ro')

    pause(60)
