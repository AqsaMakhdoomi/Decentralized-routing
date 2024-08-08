

import numpy as np
length_grid=577
rows_grid=24  #sqrt(length_grid-1)
cols_grid=24
grid=np.arange(1, length_grid, 1, dtype=int)
grid=grid.reshape((cols_grid, rows_grid))


res_G = np.load('C:\\Users\\Dell\\Downloads\\Route recomm env predicted data\\NY2km1folder\\Ny_data_opt_veh_cou\\24\\0\\res_G.npy', allow_pickle=True).tolist()
target_G = np.load('C:\\Users\\Dell\\Downloads\\Route recomm env predicted data\\NY2km1folder\\Ny_data_opt_veh_cou\\24\\0\\target_G.npy', allow_pickle=True).tolist()
res_D = np.load('C:\\Users\\Dell\\Downloads\\Route recomm env predicted data\\NY2km1folder\\Ny_data_opt_veh_cou\\24\\0\\res_D.npy', allow_pickle=True).tolist()
target_D = np.load('C:\\Users\\Dell\\Downloads\\Route recomm env predicted data\\NY2km1folder\\Ny_data_opt_veh_cou\\24\\0\\target_D.npy', allow_pickle=True).tolist()


res_D=np.array(res_D)
#print(res_D)
res_D=res_D.ravel()


res_G=np.array(res_G)
#print(res_D)
res_G=res_G.ravel()


print(type(target_G))

time_l=[]
folder=[]
for x in range(0,1247):
 time_instant=np.load('C:\\Users\\Dell\\Downloads\\Route recomm env predicted data\\NY2km1folder\\Ny_data_opt_veh_cou\\24\\' +str(x) +'/time.npy', allow_pickle=True).tolist()
 print(time_instant)
 print(time_instant[0])
 time_l.append(time_instant[0])
 folder.append(x)
 print(len(time_l))



res_G = np.asarray(res_G, dtype=np.float32)

res_G = res_G.reshape(576,576)


target_G = np.asarray(target_G, dtype=np.float32)
target_G = target_G.reshape(576,576)
#np.where(target_G>0)

map_f=zip(time_l,folder)
map_f=dict(map_f)  #maps (0,0) to 0
print(map_f)
map_r=zip(folder,time_l)
map_r=dict(map_r)  #maps (0,0) to 0
print("map_r",map_r)
#map_r[697]
#24


#Mapping
relative=[]
for i in range(0,cols_grid):
  for j in range(0,rows_grid):
    relative.append(tuple([i,j]))
positional=list(range(0, rows_grid*cols_grid+1,1))
map=zip(relative,positional)
map=dict(map)  #maps (0,0) to 0
print(map)
map1=zip(positional,relative)  #maps 0 to (0,0)   i.e maps grid cells to coordinate
map1=dict(map1)
print(map1)

sav=res_G

print("map_r",map_r)

def req_in_path(path):   #check detour ratiio
  #use actual req set and see how many request does current path cover
  # we need to know requests origin and destination
  rq_p=[]
  rq_comp=[]
  for i in range(0,len(path)):
    for j in range(i,len(path)):
      if target_G[path[i]][path[j]]:
        rq_p.append(tuple([path[i],path[j]]))
  for i in rq_p:
    a,b=i
    if a==b:
      rq_comp.append(tuple([a,b]))

    if a!=b:
     if ((path.index(b)-path.index(a))/len_sp(map1[a],map1[b]))<=random.uniform(1,2):
      rq_comp.append(tuple([a,b]))
  return rq_comp

def compatible_reqs(path,req_path):
  capacity=2
  edges={}
  taken_reqs=[]

  for first, second in zip(path, path[1:]):
     edges[(first, second)] = 0 #initially all edges have 0 requests
  br=0
  temp=[]
  temp2=[]
  reqs_s=[]
  cv=[]
  for c in req_path:
      temp.append(math.sqrt((c[1]-c[0])**2))
      cv.append(math.sqrt((c[1]-c[0])**2))
  temp2=temp
  temp2.sort(reverse=True)
  arr=np.argsort(np.array(temp))
  brr=list(arr[::-1])
  if brr:
     for k in brr:
      reqs_s.append(req_path[k])
  for i in reqs_s:
   br=0
   a,b=i
   ind1=path.index(a)
   ind2=path.index(b)
   for first, second in zip(path[ind1:ind2+1], path[ind1+1:ind2+1]):
      edges[(first,second)]=edges[(first,second)]+target_G[path[ind1]][path[ind2]]
   for first, second in zip(path[ind1:ind2+1], path[ind1+1:ind2+1]):
    if edges[(first,second)]>capacity:
      br=1
      for first, second in zip(path[ind1:ind2+1], path[ind1+1:ind2+1]):
       edges[(first,second)]=edges[(first,second)]-target_G[path[ind1]][path[ind2]]

   if br==0:  #req can be taken
    taken_reqs.append(i)
    m,n=i
    target_G[m][n]=0  #as all reqs are taken #target_G[m][n]-capacity

  return taken_reqs


def fare_cal(taken_reqs):
  fare=[]
  bf=2.55 #base fare
  p_m=0.35
  p_mile=1.75
  far=0
  fr=0
  mf=7
  #1 litre=1.01 USD can cover 12.5km
  #2.5km can be done in 0.2 litres which implies 0.202 $
  #15 minutes for 2.5km
  #avg speed=3mph   in northnumber avenue #https://www.forbes.com/sites/carltonreid/2019/05/22/uber-data-reveals-motoring-slower-than-walking-in-many-cities/?sh=5c35f71c16fb
  #0.129 petrol per mile as 2.5km is 1.55 miles
  #.129*1.01 is price of petrol for .129 litres
  if len(taken_reqs)>1:
   for i in taken_reqs:

    dist=(len_sp(map1[i[0]],map1[i[1]])+1)*1.55
    time=dist/0.05 #3 miles per hour is 0.05 miles per minute
    fr=bf+p_m*time +p_mile*dist
    if fr<mf:
     far=mf
    else:
     far=fr
    fare.append(0.8*far)
  else:
   dist=(len_sp(map1[taken_reqs[0][0]],map1[taken_reqs[0][1]])+1)*1.55
   time=dist/0.05
   fr=bf+p_m*time +p_mile*dist
   if fr<mf:
     far=mf
   else:
     far=fr
   fare.append(far)
  return sum(fare)/len(fare)

from matplotlib import pyplot as plt
def lorenzcurve(X):
    X_lorenz = X.cumsum() / X.sum()
    X_lorenz = np.insert(X_lorenz, 0, 0)
    X_lorenz[0], X_lorenz[-1]
    return X_lorenz

def dist_of_path(path,distances,dag,dest):
  a=list(zip(*np.where(dag==dest)))

  if path:
    return (distances[a[0][0]][a[0][1]]+len_sp(path[0],path[len(path)-1]))*1.55 +1.55
  else:#path ends at same grid cell
   return (distances[a[0][0]][a[0][1]])*1.55 +1.55#*1.75  #each grid cell is 2.5 km


def normalize(array):

  if np.min(array) < 0:
    array -= np.min(array)
  return array


def categorize_area():
  requests_grid=[]
  target_grid=[]
  area_sparse={}
  k=1
  for i in range(0,576):
    requests_grid.append(sum(res_G[i]))
    target_grid.append(sum(target_G[i]))
  area_s=zip(relative,target_grid)
  area_s=dict(area_s)
  for i in range(0,24):
    for j in range(0,24):
      reqs_total=0
      for a in range(0,k):
        for b in range(0,k):
          if area_s.get((i+a,j+b))!=None:
            reqs_total= reqs_total+ area_s[(i+a,j+b)]

      for a in range(1,k):
        for b in range(0,k):
          if area_s.get((i-a,j+b))!=None:
            reqs_total= reqs_total+ area_s[(i-a,j+b)]
      for a in range(0,k):
        for b in range(1,k):
          if area_s.get((i+a,j-b))!=None:
            reqs_total= reqs_total+ area_s[(i+a,j-b)]
      for a in range(1,k):
        for b in range(1,k):
          if area_s.get((i-a,j-b))!=None:
           reqs_total= reqs_total+ area_s[(i-a,j-b)]
      if reqs_total<=0:
         area_sparse[(i,j)]=True
      else:
        area_sparse[(i,j)]=False
  return area_sparse,area_s
area_sparse,area_s=categorize_area()  #area is considered as sparse if its 3 hop neighbors dont contain any request
#it is used for making non-myopic decision



import random
def WT(path,reqs):
  wt=[]
  for i in reqs:
    s,d=i
    t_s=path.index(s)-random.randint(int(path.index(s)*3/4), path.index(s))

    wt.append(t_s*1.55/0.05) #-#path[0]
#we assume driver is at index 0
  return wt

def cal_num(path,start,end):
  start_index = path.index(start)
  end_index=path.index(end)
  return abs(end_index-start_index)

def len_sp(source,dest):
 a,b=source
 c,d=dest
 shortest_path=[source]
 row=abs(c-a) #move these many rows down
 col=abs(d-b)
 if c!=a:
  row_dir=int(row/(c-a))
 else:
  row_dir=1
 if d!=b:
  col_dir=int(col/(d-b))
 #find min(row,col)
 else:
   col_dir=1
 if row<=col:
   eql_move=row
   #row=0   #we dont need to move any more rows now
 else:
   eql_move=col
  # col=0
 row=row-eql_move
 col=col-eql_move
 for i in range(1,eql_move+1): #adds (2,2),(3,3),...
  a=a+row_dir
  b=b+col_dir
  shortest_path.append(tuple([a,b]))

 for j in range(0,col):
   b=b+col_dir
   shortest_path.append(tuple([a,b]))
 for j in range(0,row):
   a=a+row_dir
   shortest_path.append(tuple([a,b]))

 length=len(shortest_path)-1
 return length


def time_gr(loc1,loc2):
  dist=len_sp(loc1,loc2)
  dist=dist*1.55   #miles
  tim=dist/3 #time in hours
  tim=tim*60
  return math.ceil(tim)

import sys
def findMax(arr):
    max = arr[0]
    n=len(arr)  #number of rows

    # Traverse array elements from second
    # and compare every element with
    # current max
    for i in range(1, n):
        if arr[i] > max:
            max = arr[i]
    return max


#calculates shortest path
def SP(source,des):

 a,b=source
 c,d=des
 shortest_path=[source]
 row=abs(c-a) #move these many rows down
 col=abs(d-b)
 if c!=a:
  row_dir=int(row/(c-a))
 else:
  row_dir=1
 if d!=b:
  col_dir=int(col/(d-b))
 #find min(row,col)
 else:
   col_dir=1
 if row<=col:
   eql_move=row
   #row=0   #we dont need to move any more rows now
 else:
   eql_move=col
  # col=0
 row=row-eql_move
 col=col-eql_move
 for i in range(1,eql_move+1): #adds (2,2),(3,3),...
  a=a+row_dir
  b=b+col_dir
  shortest_path.append(tuple([a,b]))

 for j in range(0,col):
   b=b+col_dir
   shortest_path.append(tuple([a,b]))
 for j in range(0,row):
   a=a+row_dir
   shortest_path.append(tuple([a,b]))


 return shortest_path



def DAG(res_G,source,capacity):  #will take as input full request(predicted) graph and generate DAG for current source
 #source is of form row index col index like (2,7)
 path=[]
 arr=[]
 dest_arr=[]
 br=0  #variable to break out of loop
 dag_arr=-100*np.ones((576, 576))

 sx,sy=source # x and y coordinates of source
 src=map[source]
 tru=0
 if sum(np.array(res_G[src]))==0: # or tru==1: #if there are no requests from current point to dest relocate drievr through shortest path to demand aware location
  val=dist(sx,sy,gr)
  #val conatins src and dest
  #val_s,val_d=val
 #move from shortest path from src to val and update src
  path=SP(source,val)
  src=map[val]
  sx,sy=val # x and y coordinates of source
 dest=0
 res_G[src][src]=0  #cant loop over src
 dest=findMax(np.array(res_G[src]))

 dest=np.where(res_G[src]==dest)[0][0]
 tempor2=min(capacity,res_G[src][dest])
 res_G[src][dest]=res_G[src][dest]- tempor2

 capacity=capacity-tempor2
 destination=map1[dest]
 dx,dy=destination
 #create DAG between source and destination
 r=sx-dx
 c=sy-dy

 for i in range(0,abs(r)+1):
  for j in range(0,abs(c)+1):
   if r!=0:
    dx=int(r/abs(r)) #direction to move
   else:
     dx=0
   if c!=0:
    dy=int(c/abs(c)) #direction to move
   else:
     dy=0
   arr.append(tuple([int(sx-dx*i),int(sy-dy*j)]))


 for f in arr:
     mp=map[f]
     x,y=f
     if (x-dx,y) in arr:
       dag_arr[map[(x,y)]][map[(x-dx,y)]]=res_G[map[(x,y)]][map[(x-dx,y)]]
     if (x,y-dy) in arr:
       dag_arr[map[(x,y)]][map[(x,y-dy)]]=res_G[map[(x,y)]][map[(x,y-dy)]]
     if (x-dx,y-dy) in arr:
       dag_arr[map[(x,y)]][map[(x-dx,y-dy)]]=res_G[map[(x,y)]][map[(x-dx,y-dy)]]
 return dag_arr,arr,r,c,destination,capacity,src,path


#find shortest path from all vertices to destination
alpha=1.5
sps=np.zeros(length_grid) #generates requests matrix of length length_grid*length_grid. Each element of matrix is randomly generated between 0 and 6
def sp(dest):
  a,b=dest
  for i in range(0,length_grid-1):
      sps[i]=len_sp(map1[i],dest)
  return sps

alpha=1.5

import math
def dp(r,c,src,dag,vertices,req_t,distances,capacity):

  req=[]
  reqs=[]
  for i in range(0,abs(r)+1):
   for j in range(0,abs(c)+1):
    req=[]
    reqs=[]  #temporary variables
    if i==0 and j==0:
      req_t[0][0]=0
      vertices[0][0]=0
      distances[0][0]=0
    else:
     if i-1>=0:
      x=0
      x=alpha*sps[src]-sps[dag[i][j]] -distances[i-1][j]
      if x>=1:
        req.append(dag[i-1][j])
        reqs.append(res_G[dag[i-1][j]][dag[i][j]] + req_t[i-1][j])  #append req_t here

     if j-1>=0:
      x=0
      x=alpha*sps[src]-sps[dag[i][j]] -distances[i][j-1]#[dag[i][j-1]]
      if x>=1:
        req.append(dag[i][j-1])
        reqs.append(res_G[dag[i][j-1]][dag[i][j]] + req_t[i][j-1])
     if i-1>=0 and j-1>=0:
      x=0
      x=alpha*sps[src]-sps[dag[i][j]] -distances[i-1][j-1]#[dag[i-1][j-1]]
      if x>=1:
        req.append(dag[i-1][j-1])
        reqs.append(res_G[dag[i-1][j-1]][dag[i][j]] + req_t[i-1][j-1])
     if req:
      mx=max(reqs)
      ind=reqs.index(mx)
      el=req[ind]
      vertices[i][j]=el
      tempor=min(capacity,res_G[el][dag[i][j]])
      res_G[el][dag[i][j]]= res_G[el][dag[i][j]]-tempor
      capacity=capacity-tempor
      z=list(zip(*np.where(dag==el)))
      a=z[0][0]
      b=z[0][1]
   #  print("a",a)
      req_t[i][j]=mx
      distances[i][j]= distances[a][b]+1
     else:  #make this vertex unreachable
       req_t[i][j]=-100
       distances[i][j]= 100
       vertices[i][j]=-100
  return vertices,req_t,distances,capacity



#initial values of these parameters for n drivers
n=30
k=5 #check the value of k from paper
u_n=[0]*(n)
pass_density=[0]*(n)
vacant_time=[0]*(n)
ratings=[0]*(n)

w1 =[-5.2]*(n)
w2 = [2.0]*(n)
w3 = [1.5]*(n)
w4 = [3.6]*(n)

def normalization(vect):
 #vect should be a list
 minimum=min(vect)
 maximum=max(vect)
 if maximum!=minimum:
  for i in range(0,len(vect)):
   vect[i]=vect[i]-minimum/(maximum-minimum)
 return vect

#utility will be increased like in previous case but it will not be utility per hour but simply summation for whole days
#here we calculate passenger density
import math
def calculate_passenger_density():
 for i in drivers:
  x,y= map1[loc[i]]

  l=0
  rq_l=[]
  dist=[]
  sum_reqs=0

  while True:
   for a,b in range(-1-l,2+l):
      sum_reqs=sum_reqs+target_D[map(x+a,y+b)]
      if target_D[map(x+a,y+b)]>0:
        rq_l.append([map(x+a,y+b)])
   if sum_reqs>=k:
    #select k requests randomly within those grid cells and calculate their distance from drivers starting point
     for c_i in rq_l[:k]:
       m,n=map1[c_i]
       dist.append(math.sqrt((x+a-x)^2+(y+b-y)^2))
     pass_density[i]=sum(dist)/len(dist)
     l=0
     break
   else:
    l=l+1
    sum_reqs=0

 return pass_density

 pass_density=calculate_passenger_density()

#now we will calculate ratings
import random

# Function to generate random taxi ratings
def generate_taxi_ratings(num_taxi):##n is number of drivers
    ratings = []
    # Generate 70% of ratings as 3 or 4
    num_high_ratings = int(0.7 * num_taxi)
    high_ratings = random.choices([3, 4], k=num_high_ratings)
    ratings.extend(high_ratings)

    # Generate the remaining 30% of ratings as 1, 2, or 5
    num_low_ratings = num_taxi - num_high_ratings
    low_ratings = random.choices([1, 2, 5], k=num_low_ratings)
    ratings.extend(low_ratings)

    return ratings

n=30
# Generate taxi ratings
ratings = generate_taxi_ratings(n) # n is number of drivers

utility=[20]*(n)

utility=normalization(utility)
passengers=normalization(pass_density)
vacant_time=normalization(vacant_time)
ratings=normalization(ratings)

utility = [a * b for a, b in zip(w1, u_n)]
passengers = [a * b for a, b in zip(w2, pass_density)]
vacant_time = [a * b for a, b in zip(w3, vacant_time)]
ratings = [a * b for a, b in zip(w4, ratings)]

priority= [a + b + c + d for a, b, c, d in zip(utility, passengers, vacant_time, ratings)]
print(priority)
priority_indices=sorted(range(len(priority)), key=lambda k: priority[k])
print(priority_indices)

top=[]
middle=[]
low=[]
u_sorted=u_n
u_sorted.sort()
for i in u_sorted[:int(0.25*n)]:
 low.append(u_n.index(i))
for i in u_sorted[int(0.25*n):int(0.75*n)]:
 middle.append(u_n.index(i))

for i in u_sorted[int(0.75*n):]:
 top.append(u_n.index(i))



def calc_reqs_threshold(a,b,n=1):
  reqs_adj=[]
  adjacent_positions=[]
  for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            # Skip the central position (a, b)
            if i == 0 and j == 0:
                continue

            # Calculate the adjacent position
            new_a = a + i
            new_b = b + j
            # Add the position to the list if it's valid
            if new_a >= 0 and new_b >= 0:
                adjacent_positions.append((new_a, new_b))
  try:
   for i in adjacent_positions:
        reqs_adj.append(res_D[map[i]])
  except:
      ap=1
  avg_reqs=sum(reqs_adj)
  avg_prob=avg_reqs/8  # 8 hops
  threshold=2*avg_prob/3
  return adjacent_positions,reqs_adj,threshold

def findpos(adjacent_pos,reqs,str):
   if str=='high':
    min_reqs=min(i for i in reqs if i > 0)
    ind=reqs.index(min_reqs)
    return adjacent_pos[ind]
   elif str=='low':
    max_reqs=max(reqs)
    ind=reqs.index(max_reqs)
    return adjacent_pos[ind]



##route recomm
def cal_path(i,loc):  # i is index of driver
 path=[]
 print("loc",loc[i])
 x,y=loc[i]
 for j in range(0,6):

  path.append(map[(x,y)])
  adjacent_pos,reqs,threshold=calc_reqs_threshold(x,y,4)
  threshold=abs(threshold)
  rnd=random.randint(0,len(reqs))
  if i in low:

   x,y=findpos(adjacent_pos,reqs,'low')
  elif i in middle:
    while (reqs[rnd]<threshold):#as long as reqs is less than threshols loop and break out of loop when threshold is met
      rnd=random.randint(0,len(reqs))

    x,y=adjacent_pos[rnd]
  else:
    try:
     x,y=findpos(adjacent_pos,reqs,'high')
    except:
      break

  res_D[map[(x,y)]]=res_D[map[(x,y)]]-2
 return path

import math
def dist(cpx,cpy,gr):
 dist=[]
 cor=[]
 for i,j in zip(gr[0],gr[1]):
  if i!=j:  #dont loop over same grid cell
    x,y=map1[i]
    dist.append(tuple([math.sqrt((cpx-x)**2+(cpy-y)**2)]))
    cor.append(tuple([x,y]))


 d_m=min(dist)
 index=dist.index(d_m)
 val= cor[index]
 return val

n=30

loc_x=np.random.randint(0, high=6, size=n, dtype=int)
loc_y=np.random.randint(0, high=6, size=n, dtype=int)
loc= list(zip(loc_x,loc_y))




def loc_sparsity_check(res_list,com_reqs,taken_reqs_matrix,target_G1):
  for i,j in com_reqs:
    taken_reqs_matrix[i][j]= taken_reqs_matrix[i][j]+target_G1[i][j]  ###1?
  return taken_reqs_matrix

def find_elements_less_than_one_third_sorted(arr):
    # Flatten the 2D array into a 1D array
    flat_arr = arr.copy()

    if len(flat_arr) == 0:
        return []  # No positive integers found

    flat_arr.sort()  # Sort the array

    index = len(flat_arr) // 3  # Calculate the 1/3rd position index

    if index >= len(flat_arr):
        return []  # Index out of range

    return flat_arr[:1000]


def find_k_elements(la,loc_sparsity_arr,x,y,k):
 #   print("x,y",x,y)
    xarr=[]
    yarr=[]
    da=la.copy()
   # print("len(da)",len(da))
    da.sort()
  #  print("da",da)
    da=da[:k]
    az=0

    a=la.copy()
    b=da.copy()
  #  print("b",b)
    az = 0
    ind = 0
    dup = {}
    indices = []
    for i in b:
        # print(i in dup)
        if i in dup:
            ind = dup[i]
            ind = ind + a[ind:].index(i) + 1  # Search for the element 'i' in the remaining portion of 'a'
            # print("ind",ind-1)
            indices.append(ind - 1)
            #     ind=ind+1
            dup[i] = ind
            az = az + ind  # Update the starting index for the next search
        # print("st_i", dup[i])
        else:
            ind1 = a.index(i)  # Search for the element 'i' in the remaining portion of 'a'
            #   print("i,ind1",i,ind1)

            dup[i] = ind1 + 1
            indices.append(ind1)
    #       print("dup",dup)
    #print("indices", indices)
    for i in indices:
     #   print("i",i,len(x),len(indices))
        xarr.append(x[i])
        yarr.append(y[i])
  #  print("xarr,yarr", xarr, yarr)
    return xarr, yarr
'''    for i in da:
        ind=la[az:].index(i)
        az=ind+1
        xarr.append(x[ind])
        yarr.append(y[ind])'''



def loc_sparsity(reqs_matrix):



# print(reqs_matrix[0])
 loc_sparsity_arr=reqs_matrix.copy()
# print("taken_reqs>0",np.where(taken_reqs_matrix>0))


 #print("reqs_matrix>0", np.where(reqs_matrix > 0))
 for i in range(0,576):
   for j in range(0,576):
    if reqs_matrix[i][j]!=0:
      loc_sparsity_arr[i][j]=taken_reqs_matrix[i][j]/reqs_matrix[i][j]


# print("loc_sp_ar>0", np.where(loc_sparsity_arr > 0))
 x=[]
 y=[]
 for i in range(0,576):
     for j in range(0,576):
         if reqs_matrix[i][j]>0:
             x.append(i)
             y.append(j)
# print("len(x",len(x),len(y))
 la=[]
 lb=[]
 for i,j in zip(x,y):

       la.append(loc_sparsity_arr[i][j])
       lb.append(reqs_matrix[i][j])
 #print("la>0", np.where(la > 0))


 #res = [idx for idx, val in enumerate(la) if val != 0]

# printing result
 #print("Indices of Non-Zero elements : " + str(res))
# x,y=np.where(loc_sparsity_arr>0)
 #print("x,y",len(x),y)
 #for i,j in zip(x,y):
  #print(loc_sparsity_arr[i][j])
 #print("len(la)",len(la))
 elements = find_elements_less_than_one_third_sorted(np.array(la))
 k1=100


 #print("elemets less than 1/3rd", elements)
 xind,yind=find_k_elements(la,loc_sparsity_arr,x,y,k1)
 #print("xind,yind",xind,yind)
 return  elements,la,x,y,xind,yind,lb

import statistics



def f7(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

import numpy as np


def kl_divergence(p, q):
    """Calculate KL divergence between two discrete distributions."""
    epsilon = 1e-9#1e-9  # A small epsilon value to avoid division by zero
    q = np.where(q == 0, epsilon, q)  # Replace 0s in q with epsilon
    p = np.where(p == 0, epsilon, p)
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))


def driver_kl_diver(driver_loc_arr,sparse):
    a1 = []
    b1 = []
    print("driver_loc_arr",driver_loc_arr)
    for i in driver_loc_arr:
        a1.append(i / sum(driver_loc_arr))
    probabilities = np.array(a1)

    # Calculate the expected frequencies
    exp = [n / sum(sparse)] * (n)
    for i in exp:
        b1.append(i / sum(exp))
    expected_prob = np.array(b1)
    kl=kl_divergence(probabilities,expected_prob)
    print("KL Divergence driver:", kl)


def kl_rider(loc_sparsity_arr,reqs,num): #reqs are total reqs in the area it doesnt take taken requests into account
    a1 = []
    b1 = []
    sum_of_elements = np.sum(reqs)
    normalized_array = reqs / sum_of_elements
    normalized_array=normalized_array*num
    normalized_array = np.round(normalized_array)

    while np.sum(normalized_array)>num:
        jk=random.randint(0,len(normalized_array)-1)
        normalized_array[jk]=max(normalized_array[jk]-1,1)

    reqs_final=normalized_array/reqs

    kl = kl_divergence(np.array(loc_sparsity_arr), np.array(reqs_final))
    print("KL Divergence rider:", kl)





reqs_matrix=np.zeros((576,576), dtype=np.float32)
taken_reqs_matrix= np.zeros((576,576), dtype=np.float32)


n=30
n_i=32
fr=[0]*(n)
fr_s=[0]*(n)
distan=[0]*(n)
futility=[0]*(n)
fwt=[0]*(n*n_i)
fare=[0]*(n)
d=list(range(0,n))  #driver indices
d_s=list(range(0,n)) #drivers sorted by their income
#40#6 #num of iterations
o=0
fu_ci=[0]*(n_i)
fu_i=0

dense=[0]*(n)
sparse=[0]*(n)
all=[0]*(n)

ol=0
wt_avg=0
eff_avg=0
ax=5#582#238
fol=582#1158
up=[fol]*(n)
prev_up=[fol]*(n)
loc_cou=0

for cd in range(1,n_i): #it gives us requests in this time slot

   target_G = np.load('C:\\Users\\Dell\\Downloads\\Route recomm env predicted data\\NY2km1folder\\Ny_data_opt_veh_cou\\24\\' +str(int(fol)) + '/target_G.npy', allow_pickle=True).tolist()
   target_G=np.array(target_G)
   target_G=target_G.reshape(576,576)
   reqs_matrix=np.add(reqs_matrix,target_G)
   t_s=map_r[fol]
   t_s=t_s+1
   fol=map_f[t_s]





for cd in range(1,n_i): #see how 1000 drivers earn over 100 iterations
 o=0
 print("loc",loc)
 for ak in prev_up:
  target_G = np.load('C:\\Users\\Dell\\Downloads\\Route recomm env predicted data\\NY2km1folder\\Ny_data_opt_veh_cou\\24\\' +str(ak) + '/target_G.npy', allow_pickle=True).tolist()
  res_G = np.load('C:\\Users\\Dell\\Downloads\\Route recomm env predicted data\\NY2km1folder\\Ny_data_opt_veh_cou\\24\\' +str(ak) + '/res_G.npy', allow_pickle=True).tolist()
  target_G = np.asarray(target_G, dtype=np.float32)
  target_G = target_G.reshape(576,576)
  res_G = np.asarray(res_G, dtype=np.float32)
  res_G = res_G.reshape(576,576)
  np.save('C:\\Users\\Dell\\Downloads\\Route recomm env predicted data\\NY2km1folder\\Ny_data_opt_veh_cou\\25\\' +str(ak) + '/res_G.npy', res_G)#.detach().numpy())
  np.save('C:\\Users\\Dell\\Downloads\\Route recomm env predicted data\\NY2km1folder\\Ny_data_opt_veh_cou\\25\\' +str(ak) + '/target_G.npy', target_G)#.detach().numpy())

 zk=[]
 r_a=1
 driver_i=0
 for hm in priority_indices:
  capacity=2
  gr=np.where(res_G>0)
  print("hm",hm)
  print("up",up)
  if driver_i==0:
   target_G = np.load('C:\\Users\\Dell\\Downloads\\Route recomm env predicted data\\NY2km1folder\\Ny_data_opt_veh_cou\\24\\' +str(up[hm]) + '/target_G.npy', allow_pickle=True).tolist()
   res_G = np.load('C:\\Users\\Dell\\Downloads\\Route recomm env predicted data\\NY2km1folder\\Ny_data_opt_veh_cou\\24\\' +str(up[hm]) + '/res_G.npy', allow_pickle=True).tolist()
   target_G = np.asarray(target_G, dtype=np.float32)
   target_G = target_G.reshape(576,576)

   res_G = np.asarray(res_G, dtype=np.float32)
   res_G = res_G.reshape(576,576)
  else:
   target_G = np.load('C:\\Users\\Dell\\Downloads\\Route recomm env predicted data\\NY2km1folder\\Ny_data_opt_veh_cou\\25\\' +str(up[hm]) + '/target_G.npy', allow_pickle=True).tolist()
   res_G = np.load('C:\\Users\\Dell\\Downloads\\Route recomm env predicted data\\NY2km1folder\\Ny_data_opt_veh_cou\\25\\' +str(up[hm]) + '/res_G.npy', allow_pickle=True).tolist()
   target_G = np.asarray(target_G, dtype=np.float32)
   target_G = target_G.reshape(576,576)

   res_G = np.asarray(res_G, dtype=np.float32)
   res_G = res_G.reshape(576,576)

  fwait_t=[]
  elements, loc_sparsity_arr, x, y, xind, yind, loc_arr = loc_sparsity(reqs_matrix)


  src=loc[hm]
  path1=[]
  path2=[]
  path12=[]
  path=[]
  if driver_i==(n-1):
   eff_avg=sum(fu_ci)/cd
   wt_avg=sum(fwt)/(n_i*cd+hm)
  if cd!=-1 :
    src1=src
  driver_i=driver_i+1
  res_list=cal_path(hm,loc)
  for ab in res_list:
      if area_sparse[map1[ab]] == True:
          sparse[hm] = sparse[hm] + 1
      all[hm] = all[hm] + 1


  req_path=req_in_path(res_list)
  if (req_path):
   dest=map1[res_list[-1]]  #last element in path is dest


   time_skip=time_gr(loc[hm],dest)  #returns time to skip in minutes
   time_skip=math.ceil(time_skip/15)  #since time is divided in 15 minute slots
   prev_up[hm]=int(up[hm])
   up[hm]=int(map_r[up[hm]])  #give time slot it will give folder name
   up[hm]=int(up[hm]+time_skip)
   up[hm]=int(map_f[up[hm]])
   loc[hm]=dest  #will next time use their future destination
   o=o+1
   dg_a,arr,r,c,dest,capacity,src,path=DAG(res_G,src,capacity)
   dag=np.zeros((abs(r)+1,abs(c)+1))

   kl=0
   for i in range(0,abs(r)+1):
    for j in range(0,abs(c)+1):
     dag[i][j]=map[arr[kl]]
     kl=kl+1
   dag=dag.astype(int)
   req_t=np.zeros((abs(r)+1,abs(c)+1))
   vertices=np.zeros((abs(r)+1,abs(c)+1))
   distances=np.zeros((abs(r)+1,abs(c)+1))
   vertices,req_t,distances,capacity=dp(r,c,src,dag,vertices,req_t,distances,capacity)
   target_G1 = target_G.copy()

   com_reqs = compatible_reqs(res_list, req_path)
   taken_reqs_matrix = loc_sparsity_check(res_list, com_reqs, taken_reqs_matrix, target_G1)

   path=path[:7]
   if com_reqs:
      fwait_t.append(WT(res_list,com_reqs))
      s_m=0
      for i in fwait_t:
        s_m=s_m+i[0]
      avg1=s_m/len(fwait_t)

      fwt[ol]=avg1
      ol=ol+1
      fr[hm]=fr[hm]+fare_cal(com_reqs)
      fare[hm]=fare_cal(com_reqs)
      distan[hm]=distan[hm]+dist_of_path(path,distances,dag,map[dest])

      tim=dist_of_path(path,distances,dag,map[dest])/3
      fu_c=(fare_cal(com_reqs)-dist_of_path(path,distances,dag,map[dest])*(0.202/1.55))   #1.55 miles can be done in 0.202$
      #u_c is utilitarian utility
      fu_ci[cd]=fu_ci[cd]+fu_c
      futility[hm]=((r_a-1)*futility[hm] + (fu_c/tim))/r_a  #utility per hour
      capacity=2
      for vc in com_reqs:
        o_x,o_y=vc
        target_G[o_x][o_y]=0
  else:
   prev_up[hm]=int(up[hm])
   #give time slot it will give folder name
   loc[hm]=src  #will next time use their future destination
   o=o+1

  np.save('C:\\Users\\Dell\\Downloads\\Route recomm env predicted data\\NY2km1folder\\Ny_data_opt_veh_cou\\25\\' +str(up[hm]) + '/res_G.npy', res_G)#.detach().numpy())
  np.save('C:\\Users\\Dell\\Downloads\\Route recomm env predicted data\\NY2km1folder\\Ny_data_opt_veh_cou\\25\\' +str(up[hm]) + '/target_G.npy', target_G)#.detach().numpy())




 r_a=r_a+1
 s = np.array(futility)
 sort_index = np.argsort(s)
 d_s=list(sort_index)
 futility=normalize(futility)
 futility1=np.sort(futility)


 elements,loc_sparsity_arr,x,y,xind,yind,loc_arr=loc_sparsity(reqs_matrix)

loc_r=[]
for item1,item2 in zip(sparse,all):
    if item2!=0:
     loc_r.append(item1/item2)
    else:
        loc_r.append(0)
driver_kl_diver(loc_r,sparse)

lc=[]
for i in loc_sparsity_arr:
    if i>0:
        lc.append(i)

la=[]
for i,j  in zip(x,y):
       la.append(reqs_matrix[i][j])
kl_rider(loc_sparsity_arr,la,np.sum(taken_reqs_matrix))

print("Platform utility",sum(fu_ci)/len(fu_ci))
print("Waiting time",sum(fwt)/len(fwt))
