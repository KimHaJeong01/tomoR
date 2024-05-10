import numpy  as np            # NumPy 모듈을 np로 임포트.            
import open3d as o3d            # Open3D 모듈을 o3d로 임포트.(3d 적용위함)
import matplotlib.pyplot as plt # Matplotlib 모듈을 plt로 임포트.(각종 함수 포함)
import copy # copy 모듈을 임포트

yaw = np.pi / 4.0  # yaw 각도 (x축 주위 회전)
pitch = 0.0        # pitch 각도 (y축 주위 회전)
roll = 0.0         # roll 각도 (z축 주위 회전)
theta_c = np.pi / 2.0 
#constants
alpha_color = ([1., 0., 0.]) # 알파를 나타내는 색상을 빨간색으로 정의
#alpha_color = ([255, 0, 0]) # **변경 코드들 색상을 나타내는 코드를 일부 변형**(2진수-10진수로)
beta_color  = ([0., 0., 1.]) # 베타를 나타내는 색상을 파란색으로 정의
#beta_color  = ([0, 0, 255])
TC_color    = ([0., 1., 0.]) # TC를 나타내는 색상을 녹색으로 정의
#TC_color    = ([0, 255, 0])
NV_color    = ([1., 0., 1.]) # NV를 나타내는 색상을 자주색으로 정의
#NV_color    = ([255, 0, 255])

(origin, xaxis, yaxis, zaxis) = ([0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]) # 좌표축을 정의
voxel_size  = 1. #do not change this value, 복셀 크기 정의

#subroutines
def numpyToOpen3D( np_pixels, _color0): # NumPy 배열을 Open3D 포인트 클라우드로 변환하는 함수
  o3d_pixels        = o3d.geometry.PointCloud() # Open3D 포인트 클라우드 객체를 생성
  o3d_pixels.points = o3d.utility.Vector3dVector( np_pixels) # 포인트 클라우드에 점들을 할당.
  o3d_pixels.paint_uniform_color( _color0) # 포인트 클라우드를 균일한 색상으로 칠함.
  return o3d_pixels #Open3D 포인트 클라우드로 반환

def createZeroPixels(_x0, _y0, _x1, _y1): # 영행렬 픽셀을 생성하는 함수입니다.
  zero_pixels_np = np.zeros( ( _x1-_x0+1, _y1-_y0+1, 3)).astype(np.int32) # 영행렬로 이루어진 NumPy 배열을 생성
  for x in range (0, _x1-_x0+1):
    for y in range(0, _y1-_y0+1):
      zero_pixels_np[x,y,0] = _x0+x;      zero_pixels_np[x,y,1] = _y0+y; # 배열에 픽셀 좌표를 채움
  return zero_pixels_np # 영행렬 픽셀 반환

def Area2( A,B,C):#assume z-component zero ,삼각형의 면적을 계산하는 함수
  return np.cross( B-A, C-A)[2] # AB와 AC 벡터의 외적을 계산
  
#def getBoundary( _mesh0): # 메쉬의 경계를 얻는 함수
#  v0_np = np.asarray( _mesh0.vertices) # 메쉬의 정점을 NumPy 배열로 변환
#  mesh0_min = [np.min( v0_np[:,0]), np.min( v0_np[:,1]), np.min( v0_np[:,2])]  # 좌표의 최솟값을 서치
# mesh0_max = [np.max( v0_np[:,0]), np.max( v0_np[:,1]), np.max( v0_np[:,2])]  # 좌표의 최댓값을 서치
#  return (mesh0_min,mesh0_max)  # 최솟값과 최댓값을 반환

def getBoundary(_mesh0):
    v0_np = np.asarray(_mesh0.vertices)  # 메쉬의 정점을 NumPy 배열로 변환
    if len(v0_np) == 0:
        # 정점 배열이 비어있는 경우
        print("Error: The vertex array is empty.")
        return None, None
    else:
        # 정상적인 경우 최솟값과 최댓값을 계산하여 반환
        mesh0_min = [np.min(v0_np[:, 0]), np.min(v0_np[:, 1]), np.min(v0_np[:, 2])]  # 좌표의 최솟값을 서치
        mesh0_max = [np.max(v0_np[:, 0]), np.max(v0_np[:, 1]), np.max(v0_np[:, 2])]  # 좌표의 최댓값을 서치
        return mesh0_min, mesh0_max  # 최솟값과 최댓값을 반환

def triCoord( _point, _triA, _triB, _triC):#assume z-component zero ,삼각형에 대한 점의 바리센트릭 좌표를 계산하는 함수
  (x,y,z) = _point;   ax, ay, az = _triA;  bx, by, bz = _triB;  cx, cy, cz = _triC; # 좌표를 언패킹
  area0 = Area2( _triA, _triB, _triC) # 삼각형의 면적을 계산
  side_1 = (x - bx) * (ay - by) - (ax - bx) * (y - by) # 첫 번째 바리센트릭 좌표를 계산
  side_2 = (x - cx) * (by - cy) - (bx - cx) * (y - cy) # 두 번째 바리센트릭 좌표를 계산
  side_3 = (x - ax) * (cy - ay) - (cx - ax) * (y - ay) # 세 번째 바리센트릭 좌표를 계산
  return (side_2 / area0, side_3/area0, side_1/area0) # 바리센트릭 좌표를 반환

def triPixel( tri1): # 삼각형 내부의 픽셀을 계산하는 함수.
  tri0 = copy.deepcopy(tri1); tri0[:,2]=0; # 삼각형의 복사본을 만들고 z-좌표를 0으로 설정
  mesh1_min = [np.min( tri1[:,0]), np.min( tri1[:,1]), np.min( tri1[:,2])] # 좌표의 최솟값을 서치
  mesh1_max = [np.max( tri1[:,0]), np.max( tri1[:,1]), np.max( tri1[:,2])] # 좌표의 최댓값을 서치
  (x1,y1,z1) = list(map(int, mesh1_max)) # 최댓값을 정수로 변환
  (x0,y0,z0) = list(map(int, mesh1_min)) # 최솟값을 정수로 변환
  x = np.arange( x0, x1+1).astype(np.int32) # x-좌표 배열을 만듬
  y = np.arange( y0, y1+1).astype(np.int32) # y-좌표 배열을 만듬
  vox_coord  = np.vstack((np.repeat(x, len(y)), np.tile(y, len(x)))).T.reshape( len(x), len(y), 2) # (integer),복셀 좌표를 만듬
  vox_data =  np.empty((0,3), float)# return data, 복셀 데이터를 저장할 빈 배열을 만듬
  for i in range(len(x)):
    for j in range(len(y)):
      v_center =  np.array( [vox_coord[i,j,0] ,vox_coord[i,j,1], 0] )  # 복셀의 중심을 구합
      (u,v,w) = triCoord( v_center, tri0[0], tri0[1], tri0[2]) # 바리센트릭 좌표를 계산
      if u > -0.01 and v > -0.01 and w > -0.01: # 점이 삼각형 내부에 있는지 확인
        h = tri1[0,2] * u + tri1[1,2] * v + tri1[2,2] * w # 셀의 높이를 계산
        vox_data = np.append( vox_data, np.array([[vox_coord[i,j,0],vox_coord[i,j,1],h]]) , axis=0) # 복셀 데이터를 추가
  return vox_data # 셀 데이터를 반환

def pixelizeMesh( _mesh0,_color0): # 메쉬를 픽셀화하는 함수
  vtx_np = np.asarray( _mesh0.vertices) # 메쉬의 정점을 NumPy 배열로 변환
  ele_np = np.asarray( _mesh0.triangles) # 메쉬의 삼각형을 NumPy 배열로 변환
  pixels_np = np.empty((0,3), float) # 픽셀 데이터를 저장할 빈 배열을 만듬
  for ele in ele_np:
    tri = np.array([vtx_np[ele[0]], vtx_np[ele[1]], vtx_np[ele[2]] ]) # 삼각형을 배열로 만듬
    pixels_np = np.append( pixels_np, triPixel( tri), axis=0) # 삼각형 내부의 픽셀을 계산하여 배열에 추가
  pixels_o3d = numpyToOpen3D( pixels_np, _color0 ) # 픽셀 데이터를 Open3D 포인트 클라우드로 변환
  return ( pixels_np, pixels_o3d) # 픽셀 데이터와 Open3D 포인트 클라우드를 반환

def selectiveMeshCopy(_mesh0, _bMaskList, _color): # 메시를 선택적으로 복사하는 함수
  mesh1 = copy.deepcopy(_mesh0)  # 메시의 깊은 복사본을 생성
  mesh1.triangles        = o3d.utility.Vector3iVector( np.asarray( _mesh0.triangles)[ _bMaskList] )  # 선택된 삼각형을 포함하는 새로운 메시를 생성
  mesh1.triangle_normals = o3d.utility.Vector3dVector( np.asarray( _mesh0.triangle_normals)[ _bMaskList] ) # 선택된 삼각형의 법선을 포함하는 새로운 메시를 생성
  mesh1.paint_uniform_color( _color) # 새로운 메시를 균일한 색으로 칠합
  return mesh1 # 새로운 메시를 반환

def createTopCoverPixels( _mapToAdd0, _x0, _y0, _x1, _y1, _color0): # 위쪽 커버 픽셀을 생성하는 함수
  tc_pixels_np2d = createZeroPixels( _x0, _y0, _x1, _y1) # 영행렬 픽셀을 생성
  _mapToAdd = np.unique( copy.deepcopy(_mapToAdd0), axis=0) #remove redundant pixels, 중복되는 픽셀을 제거
  for pixel in _mapToAdd:
    (a_x, a_y, a_z) = pixel.astype(np.int32)
    if tc_pixels_np2d[ a_x - x0, a_y - y0, 2] < a_z:
      tc_pixels_np2d[  a_x - x0, a_y - y0, 2] = a_z
  tc_pixelheights_for_plot = [val[:,2] for val in tc_pixels_np2d]
  tc_pixels_np = np.reshape( copy.deepcopy(tc_pixels_np2d), (-1,3)) # 렌더링을 위해 배열 크기를 변경
  tc_pixels_o3d= numpyToOpen3D( tc_pixels_np, _color0 ) # NumPy 배열을 Open3D 포인트 클라우드로 변환
  return (tc_pixelheights_for_plot, tc_pixels_np, tc_pixels_o3d) # 결과를 반환

def createPlot( _mapToAdd0, _x0, _y0, _x1, _y1): # 플롯을 생성하는 함수
  pixels_np2d = createZeroPixels( _x0, _y0, _x1, _y1) # 영행렬 픽셀을 생성
  _mapToAdd = np.unique( copy.deepcopy(_mapToAdd0), axis=0)  # 중복되는 픽셀을 제거
  for pixel in _mapToAdd:
    (a_x, a_y, a_z) = pixel.astype(np.int32)
    pixels_np2d[  a_x - x0, a_y - y0, 2] += a_z
  pixelheights_for_plot = [val[:,2] for val in pixels_np2d] # 픽셀 높이를 플롯에 사용할 수 있도록 만
  return pixelheights_for_plot # 결과를 반환

filename = "C:\\Users\\inhwa\\Contacts\\Desktop\\python files\\bunny_0.0.0.obj" #You should upload file to Google Drive manually, in advance.

# Step 1. open mesh data
#print('Loading input mesh data') 
#mesh0 = o3d.io.read_triangle_mesh(filename) # 파일에서 삼각형 메쉬 데이터를 읽어
#mesh0 = mesh0.translate( mesh0.get_center() * -1.) # 메쉬를 중심으로 이동
#mesh0.compute_vertex_normals() # 메쉬의 정점 법선을 계산
#(mesh0_min, mesh0_max) = getBoundary( mesh0) # 메쉬의 경계를 가져옴
#parameters
#theta_c = np.pi / 2.  # critical angle, in radian
#(yaw, pitch, roll) = ( np.pi /4., 0, 0) #orientation, in radian

# Step 1. open mesh data
print('Loading input mesh data') 
mesh0 = o3d.io.read_triangle_mesh(filename) # 파일에서 삼각형 메쉬 데이터를 읽어옵니다.
if mesh0 is None or len(mesh0.vertices) == 0:
    # 만약 메시가 비어있거나 읽을 수 없는 경우 오류 메시지를 출력하고 프로그램 종료
    print("Error: Unable to load mesh data or the mesh is empty.")
    exit()

mesh0 = mesh0.translate(mesh0.get_center() * -1.) # 메쉬를 중심으로 이동
mesh0.compute_vertex_normals() # 메쉬의 정점 법선을 계산
(mesh0_min, mesh0_max) = getBoundary(mesh0) # 메쉬의 경계를 가져옴
if mesh0_min is None or mesh0_max is None:
    # 만약 메쉬의 경계를 가져올 수 없는 경우 오류 메시지를 출력하고 프로그램 종료
    print("Error: Unable to get boundary of the mesh.")
    exit()

# Step 2. rotate and move onto bottom plate
mesh1 = copy.deepcopy(mesh0)  # 메쉬를 복사
R = mesh0.get_rotation_matrix_from_xyz((yaw, pitch, roll)) # 회전 행렬을 생성
mesh1.rotate( R, center=(0,0,0)) # 메쉬를 회전
v_np = np.asarray( mesh1.vertices)  # 메쉬의 정점을 가져옴
(mesh1_min, mesh1_max) = getBoundary( mesh1) # 메쉬의 경계를 가져옴
mesh1 = mesh1.translate( np.asarray(mesh1_min) * -1.) # 메쉬를 이동
mesh1.compute_vertex_normals() # 메쉬의 정점 법선을 다시 계산

v_np = np.asarray( mesh1.vertices) # 메쉬의 정점을 다시 가져옴
(mesh1_min, mesh1_max) = getBoundary( mesh1) # 메쉬의 경계를 다시 가져옴
(x1,y1,z1) = list(map(int, mesh1_max)) # 메쉬의 최대 값 좌표를 가져옴
(x0,y0,z0) = list(map(int, mesh1_min)) # 메쉬의 최소 값 좌표를 가져옴

# Step 3. find alpha, beta, NV, TC pixels
print('Finding alpha, beta, NV pixels') 
n_np = np.asarray( mesh1.triangle_normals) # 메쉬의 삼각형 법선을 가져옴
bAlpha = np.dot( n_np, zaxis) >  0.01  # 알파 픽셀을 서치
bBeta  = np.dot( n_np, zaxis) < -0.01  # 베타 픽셀을 서치
bNV    = np.logical_and( bBeta, np.dot( n_np, zaxis) > - np.sin( theta_c ) )  # NV 픽셀을 서

alpha_mesh_o3d = selectiveMeshCopy( mesh1, bAlpha, alpha_color) # 알파 메쉬를 생성
beta_mesh_o3d  = selectiveMeshCopy( mesh1, bBeta,  beta_color) # 베타 메쉬를 생성
NV_mesh_o3d    = selectiveMeshCopy( mesh1, bNV,    NV_color)  # NV 메쉬를 생성

(alpha_pixels_np, alpha_pixels_o3d)  = pixelizeMesh( alpha_mesh_o3d, alpha_color) # 알파 픽셀을 생성
( beta_pixels_np,  beta_pixels_o3d)  = pixelizeMesh(  beta_mesh_o3d, beta_color) # 베타 픽셀을 생성
(   NV_pixels_np,    NV_pixels_o3d)  = pixelizeMesh(    NV_mesh_o3d, NV_color)  # NV 픽셀을 생성

alpha_pixels_np = np.unique( alpha_pixels_np.astype(np.int32), axis=0)  # 중복 픽셀을 제거

(TC_plot, TC_pixels_np, TC_pixels_o3d)  = createTopCoverPixels( alpha_pixels_np, x0, y0, x1, y1,  TC_color) 

# Step 4. calculate volume
print("Volume info. for [",filename, "]")
Va  = alpha_pixels_np.sum( axis=0)[2] * (voxel_size*voxel_size) # 알파 픽셀의 부피를 계산
Vb  =  beta_pixels_np.sum( axis=0)[2] * (voxel_size*voxel_size) # 베타 픽셀의 부피를 계산
Vtc =    TC_pixels_np.sum( axis=0)[2] * (voxel_size*voxel_size) # 위쪽 커버 픽셀의 부피를 계산
Vnv =    NV_pixels_np.sum( axis=0)[2] * (voxel_size*voxel_size) # NV 픽셀의 부피를 계산
Vo  = Va - Vb; # 오브젝트 부피를 계산
Vss = - Va + Vb + Vtc - Vnv; # 그 외 부피를 계산
print('Va=', Va, ', Vb=', Vb, ', Vtc=', Vtc,  ', Vnv=', Vnv)
print('Vo=', Vo, ', Vss=', Vss)
# Step 5. rendering  tomographs 
print('Rendering tomographs..')
TC_pixels_o3d.translate( np.array([0.1, 0.1, 0.1]))   # 디스플레이를 위해 위쪽 커버 픽셀에 오프셋을 적용
NV_pixels_o3d.translate( np.array([-0.1, -0.1, 0.1])) # 디스플레이를 위해 NV 픽셀에 오프셋을 적용

alpha_plot = createPlot( alpha_pixels_np, x0, y0, x1, y1) # 알파 플롯 생성
beta_plot  = createPlot(  beta_pixels_np, x0, y0, x1, y1) # 베타 플롯 생성
NV_plot    = createPlot(    NV_pixels_np, x0, y0, x1, y1) # NV 플롯 생성
Vo_plot    = (np.array( alpha_plot) - np.array( beta_plot)).tolist() # VO 플롯 생성
SS_plot    = (-np.array( alpha_plot) + np.array( beta_plot) + np.array( TC_plot) - np.array( NV_plot)).tolist() # SO 플로 생성

plt.subplot(2,3,1);plt.title("α"); plt.imshow(alpha_plot, aspect='equal', cmap=plt.get_cmap('Reds')); #서브 플롯들 행과 열에 맞춰 생성 및 제목 설정, 이미지 표시
plt.subplot(2,3,2);plt.title("β"); plt.imshow( beta_plot, aspect='equal', cmap=plt.get_cmap('Blues'));
plt.subplot(2,3,3);plt.title("Vo"); plt.imshow(   Vo_plot, aspect='equal', cmap=plt.get_cmap('YlGn'));
plt.subplot(2,3,4);plt.title("TC"); plt.imshow(   TC_plot, aspect='equal', cmap=plt.get_cmap('Greens'));
plt.subplot(2,3,5);plt.title("NV"); plt.imshow(   NV_plot, aspect='equal', cmap=plt.get_cmap('Oranges'));
plt.subplot(2,3,6);plt.title("SS"); plt.imshow(   SS_plot, aspect='equal', cmap=plt.get_cmap('BuPu'));
plt.show()
 
#step 6. 3D plots 

#o3d.visualization.draw_geometries([alpha_mesh_o3d+beta_mesh_o3d+NV_mesh_o3d])#view triangle mesh
#o3d.visualization.draw_geometries([alpha_pixels_o3d+beta_pixels_o3d+NV_pixels_o3d+TC_pixels_o3d])

fig = plt.figure(figsize=(10, 5)) #새로운 Figure 객체 생성하고 Figure의 크기를 설정
ax = fig.add_subplot(111, projection='3d') # 3D 서브플롯을 추가하고 Axe3D 객체를 가져옴
ax.view_init(elev=20., azim=145) # 카메라의 시점을 조정하여 시각화를 보기 좋게 설정
ax.scatter( alpha_pixels_np[:,0], alpha_pixels_np[:,1], alpha_pixels_np[:,2], alpha=0.5, color='red') # 각각의 점들을 3D 공간에 산포도로 표시
ax.scatter(  beta_pixels_np[:,0],  beta_pixels_np[:,1],  beta_pixels_np[:,2], alpha=0.5, color='blue')
ax.scatter(    TC_pixels_np[:,0],    TC_pixels_np[:,1],    TC_pixels_np[:,2], alpha=0.5, color='green')
ax.scatter(    NV_pixels_np[:,0],    NV_pixels_np[:,1],    NV_pixels_np[:,2], alpha=0.5, color='magenta')
plt.title("α(Red),β(Blue),TC(green), NV(magenta)")
plt.show()


 

