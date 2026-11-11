import numpy as np 


def rotation_matrix_from_axis(angle, n_vec):
    '''
    给定角度和旋转轴输出旋转矩阵
    '''
    n = np.asarray(n_vec, dtype=float)
    n = n / np.linalg.norm(n)
    x, y, z = n
    c = np.cos(angle)
    s = np.sin(angle)
    t = 1 - c
    R = np.array([
        [t*x*x + c,     t*x*y - s*z,   t*x*z + s*y],
        [t*x*y + s*z,   t*y*y + c,     t*y*z - s*x],
        [t*x*z - s*y,   t*y*z + s*x,   t*z*z + c]
    ])
    return R

def _kabsch(P:np.ndarray, Q:np.ndarray):
    '''
    给定两组点云，输出最优旋转矩阵R和平行矢量t，使得
    (R @ P.T).T + t 与 Q 的距离最小。先旋转后平移。
    '''
    assert len(P) > 2
    centroid_P = np.mean(P, axis=0)
    centroid_Q = np.mean(Q, axis=0)
    P_centered = P - centroid_P 
    Q_centered = Q - centroid_Q
    H = P_centered.T.dot(Q_centered)   
    U, S, VT = np.linalg.svd(H) 
    R = U.dot(VT).T  
    if np.linalg.det(R) < 0:            
        VT[2,:] *= -1  
        R = U.dot(VT).T 
    t = centroid_Q - R.dot(centroid_P)   
    return R, t


def adjust_positions(pos:np.ndarray, dpos:np.ndarray):
    '''
    调整pos，使得pos跟pos+dpos尽可能接近
    '''
    if pos.ndim == 1 or len(pos) == 1:
        return pos + dpos 
    new_pos = pos + dpos 
    if len(pos) == 2:
        center     = 0.5*(pos[0] + pos[1])
        center_new = 0.5*(new_pos[0] + new_pos[1])
        vec        = pos[0] - pos[1]
        vec        = vec / np.linalg.norm(vec)
        vec_new    = new_pos[0] - new_pos[1]
        vec_new    = vec_new / np.linalg.norm(vec_new)
        n_vec = np.cross(vec, vec_new)
        if np.all(np.isclose(n_vec, 0.0)): #两个接近平行
            return pos + center_new - center
        angle = np.arccos(np.dot(vec, vec_new))
        R = rotation_matrix_from_axis(angle, n_vec)
        centered_pos = pos - center # 先让pos 绕着自己的中心旋转
        centered_pos = (R @ centered_pos.T).T
        return centered_pos + center_new
    R, t = _kabsch(pos, new_pos)
    return (R@pos.T).T + t

def rigid_body_forces(positions:np.ndarray, forces:np.ndarray, r0:np.ndarray=None)->np.ndarray:
    '''
    调整受力使原子组近似做刚体运动（因为没考虑向心加速度）
    '''
    n = len(positions)
    r_c = r0 if not r0 is None else np.mean(positions, axis=0)  # 质心
    F_total = np.sum(forces, axis=0) 
    a_cm = F_total / n
    r_rel = positions - r_c 
    tau_total = np.sum(np.cross(r_rel, forces), axis=0) 
    I = np.zeros((3, 3))
    for i in range(n):
        r = r_rel[i]
        I += (np.dot(r, r) * np.eye(3) - np.outer(r, r))
    try:
        alpha = np.linalg.solve(I, tau_total)
    except np.linalg.LinAlgError:
        alpha = np.zeros(3) 
    new_forces = a_cm + np.cross(alpha, r_rel)
    return new_forces
