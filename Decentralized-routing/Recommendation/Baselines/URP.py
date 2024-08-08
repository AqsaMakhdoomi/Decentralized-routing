import numpy as np
length_grid=577
rows_grid=24
cols_grid=24
grid=np.arange(1, length_grid, 1, dtype=int)
grid=grid.reshape((cols_grid, rows_grid))

res_G = np.load('C:\\Users\\Dell\\Downloads\\Route recomm env predicted data\\NY2km1folder\\Ny_data_opt_veh_cou\\24\\0\\res_G.npy', allow_pickle=True).tolist()
target_G = np.load('C:\\Users\\Dell\\Downloads\\Route recomm env predicted data\\NY2km1folder\\Ny_data_opt_veh_cou\\24\\0\\target_G.npy', allow_pickle=True).tolist()
res_D = np.load('C:\\Users\\Dell\\Downloads\\Route recomm env predicted data\\NY2km1folder\\Ny_data_opt_veh_cou\\24\\0\\res_D.npy', allow_pickle=True).tolist()
target_D = np.load('C:\\Users\\Dell\\Downloads\\Route recomm env predicted data\\NY2km1folder\\Ny_data_opt_veh_cou\\24\\0\\target_D.npy', allow_pickle=True).tolist()

print(type(target_G))

time_l=[]
folder=[]

print(time_l)

with open('C:\\Users\\Dell\\Downloads\\Route recomm env predicted data\\NY2km1folder\\Ny_data_opt_veh_cou\\24\\time_l.txt', 'r') as f:
    time_l = np.loadtxt(f)
with open('C:\\Users\\Dell\\Downloads\\Route recomm env predicted data\\NY2km1folder\\Ny_data_opt_veh_cou\\24\\folder_l.txt', 'r') as f:
   folder = np.loadtxt(f)

import math
hour=[0]*len(time_l)
rounded_hour=[0]*len(time_l)
day=[0]*len(time_l)
km=0
for i in time_l:
 hour[km]=math.ceil(i/4)
 rounded_hour[km]=hour[km]%24
 day[km]=math.ceil(hour[km]/24)
 km=km+1

time_l[5]

rounded_hour[5]

print("rounded_hour",rounded_hour)

def find_missing(lst):
    max = lst[0]
    for i in lst :
      if i > max :
        max= i

    min = lst [0]
    for i in lst :
      if i < min:
        min = i
    missing = max+1
    list1=[]

    for _ in lst :

        max = max -1
        if max not in lst :
          list1.append(max)

    return list1

time_l

lst = [1,5,4,6,8, 2,3, 7, 9, 10]
print(find_missing(time_l))

res_G = np.asarray(res_G, dtype=np.float32)

res_G = res_G.reshape(576, 576)

target_G = np.asarray(target_G, dtype=np.float32)
target_G = target_G.reshape(576, 576)

map_f=zip(time_l,folder)
map_f=dict(map_f)
print(map_f)
map_r=zip(folder,time_l)
map_r=dict(map_r)
print(map_r)

map_f[1657]

relative=[]
for i in range(0,cols_grid):
  for j in range(0,rows_grid):
    relative.append(tuple([i,j]))
positional=list(range(0, rows_grid*cols_grid+1,1))
map=zip(relative,positional)
map=dict(map)
print(map)
map1=zip(positional,relative)
map1=dict(map1)
print(map1)

sav=res_G

def SP(source,des):

 a,b=source
 c,d=des
 shortest_path=[source]
 row=abs(c-a)
 col=abs(d-b)
 if c!=a:
  row_dir=int(row/(c-a))
 else:
  row_dir=1
 if d!=b:
  col_dir=int(col/(d-b))

 else:
   col_dir=1
 if row<=col:
   eql_move=row

 else:
   eql_move=col

 row=row-eql_move
 col=col-eql_move
 for i in range(1,eql_move+1):
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

def len_sp(source,dest):
 a,b=source
 c,d=dest
 shortest_path=[source]
 row=abs(c-a)
 col=abs(d-b)
 if c!=a:
  row_dir=int(row/(c-a))
 else:
  row_dir=1
 if d!=b:
  col_dir=int(col/(d-b))

 else:
   col_dir=1
 if row<=col:
   eql_move=row

 else:
   eql_move=col

 row=row-eql_move
 col=col-eql_move
 for i in range(1,eql_move+1):
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

import sys
def findMax(arr):
    max = arr[0]
    n=len(arr)

    for i in range(1, n):
        if arr[i] > max:
            max = arr[i]
    return max

def dist(cpx,cpy,gr):
 dist=[]
 cor=[]

 gr_new=[]
 for a in gr[0]:
   gr_new.append(map1[a])
 for i,j in gr_new:
  if i!=j:
    dist.append(tuple([math.sqrt((cpx-i)**2+(cpy-j)**2)]))
    cor.append(tuple([i,j]))

 d_m=min(dist)
 index=dist.index(d_m)
 val= cor[index]

 print("gr",gr)
 print("gr_new",gr_new)
 print("new src",val)
 return val

import random

n=30
import random
data = list(range(0, 23))
random.shuffle(data)
loc=random.sample(range(0, 23*23), n)

capacity=2
gr=np.where(res_G>0)
print(gr)

def DAG(res_G,source,capacity,gr):

 if type(source) is not tuple:
     source=map1[source]
 path=[]
 arr=[]
 dest_arr=[]
 br=0

 dag_arr=-100*np.ones((576, 576))

 sx,sy=source
 src=map[source]
 tru=0

 if sum(np.array(res_G[src]))==0:
  val=dist(sx,sy,gr)

  path=SP(source,val)
  src=map[val]
  sx,sy=val

 dest=0
 res_G[src][src]=0
 dest=findMax(np.array(res_G[src]))

 dest=np.where(res_G[src]==dest)[0][0]

 tempor2=min(capacity,res_G[src][dest])
 res_G[src][dest]=res_G[src][dest]- tempor2

 capacity=capacity-tempor2

 destination=map1[dest]

 dx,dy=destination

 r=sx-dx
 c=sy-dy

 for i in range(0,abs(r)+1):
  for j in range(0,abs(c)+1):
   if r!=0:
    dx=int(r/abs(r))
   else:
     dx=0
   if c!=0:
    dy=int(c/abs(c))
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

 return dag_arr,arr,r,c,destination,capacity,src,path,gr

alpha=1.5
sps=np.zeros(length_grid)
def sp(dest):
  a,b=dest
  for i in range(0,length_grid-1):

      sps[i]=len_sp(map1[i],dest)
  return sps

import math

def dp(r,c,src,dag,vertices,req_t,distances,capacity):

  req=[]
  reqs=[]
  for i in range(0,abs(r)+1):
   for j in range(0,abs(c)+1):
    req=[]
    reqs=[]
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
        reqs.append(res_G[dag[i-1][j]][dag[i][j]] + req_t[i-1][j])

     if j-1>=0:

      x=0

      x=alpha*sps[src]-sps[dag[i][j]] -distances[i][j-1]
      if x>=1:
        req.append(dag[i][j-1])
        reqs.append(res_G[dag[i][j-1]][dag[i][j]] + req_t[i][j-1])
     if i-1>=0 and j-1>=0:
      x=0

      x=alpha*sps[src]-sps[dag[i][j]] -distances[i-1][j-1]
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

      req_t[i][j]=mx
      distances[i][j]= distances[a][b]+1
     else:
       req_t[i][j]=-100
       distances[i][j]= 100
       vertices[i][j]=-100
  return vertices,req_t,distances,capacity

def ret_path(vertices, src):
    dpath = [map[dest]]

    x = list(zip(*np.where(dag == map[dest])))

    a = int(vertices[x[0][0], x[0][1]])

    while a != src:
        dpath.append(a)

        x = list(zip(*np.where(dag == a)))

        a = int(vertices[x[0][0], x[0][1]])

    return dpath

def checksp(path1):
    ar = []
    path = []
    for i in path1:

        reqs = []
        coors = []
        sor_coors = []
        print("alpha", alpha)
        print("path1", path1)

        for j in range(0,length_grid-1):

         if target_G[i][j]>0:
             print("i,j", i, j,target_G[i][j])
             reqs.append(target_G[i][j])
             coors.append(j)
        reqs=np.array(reqs)
        coors=np.array(coors)
        ind=np.argsort(reqs)
        print("reqs",reqs)
        print("ind",ind)
        for ik in ind[::-1]:
         sor_coors.append(coors[ik])
        reqs=list(reqs)
        coors=list(coors)
        print("sor_cor",sor_coors)
        for j in sor_coors:
            ar.append(i)
            ar.append(j)

            print("ar",ar)

            dist_travelled=0
            print("ar",ar)
            for first, second in zip(ar, ar[1:]):
             dist_travelled=dist_travelled+len_sp(map1[first],map1[second])
             print("first,second",first,second)
             print("dist travelled",dist_travelled)
             print("len_sp(map1[ar[0]],map1[ar[-1]])", len_sp(map1[ar[0]],map1[ar[-1]]))
             if (dist_travelled)<=alpha*len_sp(map1[ar[0]],map1[ar[-1]]):
                ar = [*set(ar)]

                for first1, second1 in zip(ar, ar[1:]):
                    pt0 = SP(map1[first1], map1[second1])
                    print("1st sec in for",first1,second1)
                    for a in pt0:
                        path.append(map[a])

                print("path after appending",path)
                path=f7(path)

                if (len(path)>=count1):
                    print("leng>path")
                    return path

    return path

def cal_max(src):
    k = 1
    values = []
    coors = []
    src_x, src_y = map1[src]

    for i in range(-(k), k + 1):
        for j in range(-(k), k + 1):

            try:
             values.append(res_D[map[(src_x + i, src_y + j)]])
             coors.append(map[(src_x + i, src_y + j)])
            except:
                ap=1
    max_v = max(values)
    index = values.index(max_v)
    return coors[index]


def f7(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

def checksp(path1):
    ar = []
    path = []
    for i in path1:

        reqs = []
        coors = []
        sor_coors = []
        print("alpha", alpha)
        print("path1", path1)

        for j in range(0, length_grid - 1):

            try:
             if target_G[i][j] > 0:
                print("i,j", i, j, target_G[i][j])
                reqs.append(target_G[i][j])
                coors.append(j)
            except:
                ap=1
        reqs = np.array(reqs)
        coors = np.array(coors)
        ind = np.argsort(reqs)
        print("reqs", reqs)
        print("ind", ind)
        for ik in ind[::-1]:
            sor_coors.append(coors[ik])
        reqs = list(reqs)
        coors = list(coors)
        print("sor_cor", sor_coors)
        for j in sor_coors:
            ar.append(i)
            ar.append(j)

            print("ar", ar)

            dist_travelled = 0
            print("ar", ar)
            for first, second in zip(ar, ar[1:]):
                dist_travelled = dist_travelled + len_sp(map1[first], map1[second])
                print("first,second", first, second)
                print("dist travelled", dist_travelled)
                print("len_sp(map1[ar[0]],map1[ar[-1]])", len_sp(map1[ar[0]], map1[ar[-1]]))
                if (dist_travelled) <= alpha * len_sp(map1[ar[0]], map1[ar[-1]]):
                    ar = [*set(ar)]

                    for first1, second1 in zip(ar, ar[1:]):
                        pt0 = SP(map1[first1], map1[second1])
                        print("1st sec in for", first1, second1)
                        for a in pt0:
                            path.append(map[a])

                    print("path after appending", path)
                    path = f7(path)

                    if (len(path) >= count1):
                        print("leng>path")
                        return path

    return path

def req_in_path(path):

    rq_p = []
    rq_comp = []

    for i in range(0, len(path)):
        for j in range(i, len(path)):

            try:
             if target_G[path[i]][path[j]]:
                rq_p.append(tuple([path[i], path[j]]))
            except:
                ap=1

    for i in rq_p:
        a, b = i
        if a == b:
            rq_comp.append(tuple([a, b]))

        if a != b:
            if ((path.index(b) - path.index(a)) / len_sp(map1[a], map1[b])) <= random.uniform(1,2):
                rq_comp.append(tuple([a, b]))

    return rq_comp

def compatible_reqs(path, req_path):
    capacity = 2
    edges = {}
    taken_reqs = []

    for first, second in zip(path, path[1:]):

        edges[(first, second)] = 0
    br = 0
    temp = []
    temp2 = []
    reqs_s = []
    cv = []

    for c in req_path:

        temp.append(math.sqrt((c[1] - c[0]) ** 2))
        cv.append(math.sqrt((c[1] - c[0]) ** 2))

    temp2 = temp
    temp2.sort(reverse=True)

    arr = np.argsort(np.array(temp))
    brr = list(arr[::-1])
    if brr:
        for k in brr:

            reqs_s.append(req_path[k])

    for i in reqs_s:
        br = 0
        a, b = i
        ind1 = path.index(a)
        ind2 = path.index(b)

        for first, second in zip(path[ind1:ind2 + 1], path[ind1 + 1:ind2 + 1]):
            edges[(first, second)] = edges[(first, second)] + target_G[path[ind1]][path[ind2]]
        for first, second in zip(path[ind1:ind2 + 1], path[ind1 + 1:ind2 + 1]):

            if edges[(first, second)] > capacity:
                br = 1
                for first, second in zip(path[ind1:ind2 + 1], path[ind1 + 1:ind2 + 1]):
                    edges[(first, second)] = edges[(first, second)] - target_G[path[ind1]][path[ind2]]

        if br == 0:
            taken_reqs.append(i)
            m, n = i
            target_G[m][n] = 0

    return taken_reqs

''' BF:$2.55

$0.35 per minute + $1.75 per mile

$7.00'''

def fare_cal(taken_reqs):
  fare=[]
  bf=2.55
  p_m=0.35
  p_mile=1.75
  far=0
  fr=0
  mf=7

  if len(taken_reqs)>1:
   for i in taken_reqs:

    dist=(len_sp(map1[i[0]],map1[i[1]])+1)*1.55
    time=dist/0.05
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

  print("fare is",sum(fare))
  return sum(fare)

from matplotlib import pyplot as plt
def lorenzcurve(X):
    X_lorenz = X.cumsum() / X.sum()

    X_lorenz = np.insert(X_lorenz, 0, 0)
    X_lorenz[0], X_lorenz[-1]
    return X_lorenz

def dist_of_path(path, distances, dag, dest):
    a = list(zip(*np.where(dag == dest)))

    print("a",a)

    if path:

        try:
         return (distances[a[0][0]][a[0][1]] + len_sp(path[0],
                                                     path[len(path) - 1])) * 1.55 + 1.55
        except:
            return ( len_sp(path[0],
                                                         path[len(path) - 1])) * 1.55 + 1.55

    else:
        try:
         return (distances[a[0][0]][a[0][1]]) * 1.55 + 1.55
        except:
            return 0

''' https://www.investopedia.com/articles/personal-finance/021015/uber-versus-yellow-cabs-new-york-city.asp
BF:$2.55

$0.35 per minute + $1.75 per mile

$7.00'''

def gini(arr):

    sorted_arr = arr.copy()
    sorted_arr.sort()
    n = arr.size
    coef_ = 2. / n
    const_ = (n + 1.) / n
    weighted_sum = sum([(i+1)*yi for i, yi in enumerate(sorted_arr)])
    return coef_*weighted_sum/(sorted_arr.sum()) - const_

def normalize(array):

  if np.min(array) < 0:
    array -= np.min(array)
  return array

def categorize_area():
    requests_grid = []
    target_grid = []
    area_sparse = {}
    k = 1
    for i in range(0, 576):

        requests_grid.append(sum(res_G[i]))
        target_grid.append(sum(target_G[i]))

    area_s = zip(relative, requests_grid)
    area_s = dict(area_s)

    for i in range(0, 24):
        for j in range(0, 24):
            reqs_total = 0
            for a in range(0, k):
                for b in range(0, k):

                    if area_s.get((i + a, j + b)) != None:
                        reqs_total = reqs_total + area_s[(i + a, j + b)]

            for a in range(1, k):
                for b in range(0, k):

                    if area_s.get((i - a, j + b)) != None:
                        reqs_total = reqs_total + area_s[(i - a, j + b)]
            for a in range(0, k):
                for b in range(1, k):

                    if area_s.get((i + a, j - b)) != None:
                        reqs_total = reqs_total + area_s[(i + a, j - b)]
            for a in range(1, k):
                for b in range(1, k):

                    if area_s.get((i - a, j - b)) != None:
                        reqs_total = reqs_total + area_s[(i - a, j - b)]

            if reqs_total <= 0:
                area_sparse[(i, j)] = True
            else:
                area_sparse[(i, j)] = False

    return area_sparse, area_s

area_sparse, area_s = categorize_area()

l=[1,2]
sum(l)

def calc_rel(location, hops):
    i, j = location
    area_r = []
    req_area = []

    for a in range(0, hops):
        for b in range(0, hops):

            if area_s.get((i + a, j + b)) != None:
                if area_s.get((i + a, j + b)) > 0:
                    area_r.append((i + a, j + b))

                    req_area.append(area_s.get((i + a, j + b)))
    for a in range(1, hops):
        for b in range(0, hops):

            if area_s.get((i - a, j + b)) != None:
                if area_s.get((i - a, j + b)) > 0:
                    area_r.append((i - a, j + b))

                    req_area.append(area_s.get((i - a, j + b)))
    for a in range(0, hops):
        for b in range(1, hops):

            if area_s.get((i + a, j - b)) != None:
                if area_s.get((i + a, j - b)) > 0:
                    area_r.append((i + a, j - b))

                    req_area.append(area_s.get((i + a, j - b)))
    for a in range(1, hops):
        for b in range(1, hops):

            if area_s.get((i - a, j - b)) != None:
                if area_s.get((i - a, j - b)) > 0:
                    area_r.append((i - a, j - b))

                    req_area.append(area_s.get((i - a, j - b)))

    return area_r, req_area

def relocate(location,hops):

    in_l = 0
    area_r = []
    req_area = []

    area_r, req_area = calc_rel(location, hops)
    sorted_area = [0] * (10000)
    sorted_req_area = [0] * (10000)
    loop=0

    while len(req_area) == 0:
        area_r, req_area = calc_rel(location, hops + 1)
        loop=loop+1
        if loop==10:
          print("loop==10")
          return location

    indices = np.argsort(np.array(req_area))

    k = 0
    for i in indices[::-1]:

        sorted_area[k] = area_r[i]
        sorted_req_area[k] = req_area[i]
        k = k + 1

    for i in sorted_area:
        if area_sparse[i] == False:
            max_v = i

            in_l = in_l + 1
            return i
        elif in_l >= len(sorted_area):

            return sorted_area[0]

def calc_rel_prev(location,hops):
    i,j=location
    req_area={}
    for a in range(0,hops):
        for b in range(0,hops):

          if area_s.get((i+a,j+b))!=None:
            if area_s.get((i+a,j+b))>0:
               req_area[(i+a,j+b)]= area_s.get((i+a,j+b))
    for a in range(1,hops):
        for b in range(0,hops):

          if area_s.get((i-a,j+b))!=None:
            if area_s.get((i-a,j+b))>0:
               req_area[(i-a,j+b)]= area_s.get((i-a,j+b))
    for a in range(0,hops):
        for b in range(1,hops):

          if area_s.get((i+a,j-b))!=None:
              if area_s.get((i+a,j-b))>0:
               req_area[(i+a,j-b)]= area_s.get((i+a,j-b))
    for a in range(1,hops):
        for b in range(1,hops):

          if area_s.get((i-a,j-b))!=None:
            if area_s.get((i-a,j-b))>0:
               req_area[(i-a,j-b)]= area_s.get((i-a,j-b))
    return req_area

def relocate_prev(location,hops):

    req_area=calc_rel_prev(location,hops)
    print("reqs in hops",req_area)
    while not bool(req_area):
       req_area=calc_rel_prev(location,hops+1)

    max_v= max(zip(req_area.values(), req_area.keys()))[1]
    print("max value",max_v)

    return max_v

def order_ride(path,reqs):
  b=c

import random
def WT(path,reqs):
  wt=[]

  for i in reqs:
    s,d=i

    t_s=path.index(s)-random.randint(int(path.index(s)*3/4), path.index(s))

    wt.append(t_s*1.55/0.05)

  return wt

def cal_num(path,start,end):

  start_index = path.index(start)
  end_index=path.index(end)

  return abs(end_index-start_index)

def metrics_cal(pathh):
  orders=0
  ok=0
  el=0
  added_elem=[]
  efficiency=0
  lk=[]
  ct=0
  capacity=2
  c=capacity
  j=0
  pass_g=0
  copied_path=np.zeros(len(pathh))
  copied_path=list(copied_path)

  sum_eff=0
  for i in range(len(pathh)):
   if pathh[i]!='/':
     copied_path[j]=pathh[i]

     j=j+1
  print("copied path",copied_path)
  l_c=len(copied_path)
  i_c=1
  p_k=copied_path
  reversed_list=copied_path[::-1]
  count=0
  for iz in p_k:
   for kl in range(i_c-2,-1,-1):

     if target_G[p_k[kl]][iz] and [p_k[kl],iz] in added_elem :
       capacity=min(c,capacity+ target_G[p_k[kl]][iz])

   i_c=i_c+1
   count=count+1
   lk=reversed_list[:-count]

   for a in lk:
    if capacity>0:
      val=cal_num(copied_path,a,iz)

      if val<=alpha*len_sp(map1[iz],map1[a]):

       capacity=max(0,capacity-target_G[iz][a])
       added_elem.append([iz,a])

   pass_g=pass_g+c-capacity

   if c-2>=capacity:
     ok=ok+1
   if c-1==capacity:
     orders=orders+1

   efficiency=(c-capacity)/c

   sum_eff=sum_eff+efficiency
  pass_g=pass_g/l_c
  sum_eff=sum_eff/l_c
  print("sum_eff is",sum_eff)
  perc_ride_without=(orders*100)/l_c
  perc_ride_with=(ok*100)/l_c

  print("percentage of orders without ridesharing are",perc_ride_without)
  print("percentage of orders with ridesharing are",perc_ride_with)
  print("passengers per grid",pass_g)
  return perc_ride_with,pass_g

target_G>0

np.where(res_G>0)

def time_gr(loc1,loc2):

  res = type(loc2) is tuple
  print("loc1",loc1)
  print("loc2",loc2,res)
  if res==True:
   dist=len_sp(loc1,loc2)
  else:
      dist = len_sp(loc1, map1[loc2])

  dist=dist*1.55
  tim=dist/3
  tim=tim*60
  return math.ceil(tim)

def loc_sparsity_check(res_list,com_reqs,taken_reqs_matrix,target_G1):
  for i,j in com_reqs:

    taken_reqs_matrix[i][j]= taken_reqs_matrix[i][j]+target_G1[i][j]
  return taken_reqs_matrix

def find_elements_less_than_one_third_sorted(arr):

    flat_arr = arr.copy()

    if len(flat_arr) == 0:
        return []

    flat_arr.sort()

    index = len(flat_arr) // 3

    if index >= len(flat_arr):
        return []

    one_third_element = flat_arr[index]

    less_than_one_third = [num for num in flat_arr if num < one_third_element]

    return flat_arr[:1000]

def find_k_elements(la,loc_sparsity_arr,x,y,k):

    xarr=[]
    yarr=[]
    da=la.copy()

    da.sort()

    da=da[:k]
    az=0

    a=la.copy()
    b=da.copy()

    az = 0
    ind = 0
    dup = {}
    indices = []
    for i in b:

        if i in dup:
            ind = dup[i]
            ind = ind + a[ind:].index(i) + 1

            indices.append(ind - 1)

            dup[i] = ind
            az = az + ind

        else:
            ind1 = a.index(i)

            dup[i] = ind1 + 1
            indices.append(ind1)

    for i in indices:

        xarr.append(x[i])
        yarr.append(y[i])

    return xarr, yarr
'''    for i in da:
        ind=la[az:].index(i)
        az=ind+1
        xarr.append(x[ind])
        yarr.append(y[ind])'''

def loc_sparsity(reqs_matrix):

 loc_sparsity_arr=reqs_matrix.copy()

 for i in range(0,576):
   for j in range(0,576):
    if reqs_matrix[i][j]!=0:
      loc_sparsity_arr[i][j]=taken_reqs_matrix[i][j]/reqs_matrix[i][j]

 x=[]
 y=[]
 for i in range(0,576):
     for j in range(0,576):
         if reqs_matrix[i][j]>0:
             x.append(i)
             y.append(j)

 la=[]
 lb=[]
 for i,j in zip(x,y):

       la.append(loc_sparsity_arr[i][j])
       lb.append(reqs_matrix[i][j])

 elements = find_elements_less_than_one_third_sorted(np.array(la))
 k1=500

 xind,yind=find_k_elements(la,loc_sparsity_arr,x,y,k1)

 return  elements,la,x,y,xind,yind,lb

import statistics

import random
from fractions import Fraction
import numpy as np
def generate_fraction(*args):
    denominator = all

    numerator = dense
    return Fraction(numerator, denominator)

def f7(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

def sort_drivers(d,dense,all):

  drivers=[]
  init_loc=np.random.randint(0, high=n, dtype=int)

  loc_reqs=[]
  for item1,item2 in zip(dense,all):
    if item2!=0:
     loc_reqs.append(item1/item2)
    else:
      loc_reqs.append(0)

  while loc_reqs[init_loc]==0:
   init_loc=np.random.randint(0, high=n, dtype=int)
  prop_loc=init_loc
  count=0

  for i in range(1,100000):
   if init_loc==prop_loc:
    drivers.append(init_loc)
    count=count+1
   prop_loc=np.random.randint(0, high=n)
   while loc_reqs[prop_loc]==0:
    prop_loc=np.random.randint(0, high=n)

   if loc_reqs[init_loc]!=0:
    a_p=loc_reqs[prop_loc]/loc_reqs[init_loc]
   else:
     a_p=0
   if a_p>=1:
    init_loc=prop_loc

   else:

    r = random.random()
    if a_p>=r:
     init_loc=prop_loc

    else:
     init_loc=init_loc

  matrix2=loc_reqs

  drivers_noduplicate = f7(drivers)

  if len(drivers_noduplicate)>0:
   return drivers_noduplicate[::-1],loc_reqs
  else:
    return d,loc_reqs

def test_driver(driver_loc_arr,sparse):

    a1 = []
    b1 = []
    for i in driver_loc_arr:
        a1.append(i / sum(driver_loc_arr))

    probabilities = np.array(a1)

    exp=[n/sum(sparse)]*(n)

    for i in exp:
        b1.append(i / sum(exp))

    expected_prob = np.array(b1)

    test_statistic, p_value = chisquare(f_obs=probabilities , f_exp=expected_prob)

    print("Chi-square test statistic:", test_statistic)
    print("P-value:", p_value)

def kl_divergence(p, q):
    """Calculate KL divergence between two discrete distributions."""
    epsilon = 1e-9
    q = np.where(q == 0, epsilon, q)
    p = np.where(p == 0, epsilon, p)

    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def driver_kl_diver(driver_loc_arr,sparse):
    a1 = []
    b1 = []
    print("driver_loc_arr",driver_loc_arr)
    for i in driver_loc_arr:
        a1.append(i / sum(driver_loc_arr))

    probabilities = np.array(a1)

    exp = [n / sum(sparse)] * (n)

    for i in exp:
        b1.append(i / sum(exp))

    expected_prob = np.array(b1)
    kl=kl_divergence(probabilities,expected_prob)
    print("KL Divergence driver:", kl)

def kl_rider(loc_sparsity_arr,reqs,num):
    a1 = []
    b1 = []

    print(len(reqs))
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

fd_s=  [0, 350, 682, 334, 1158, 552, 1240, 859, 238, 1204, 457, 917,  582, 579, 247, 1176]

azxc=5
reqs_matrix=np.zeros((576,576), dtype=np.float32)
taken_reqs_matrix= np.zeros((576,576), dtype=np.float32)

n=30
n_i=5
fare=[0]*(n)
fr=[0]*(n)
fr_s=[0]*(n)
distan=[0]*(n)
futility=[0]*(n)
fwt=[0]*(n*n_i)

dense=[0]*(n)
sparse=[0]*(n)
all=[0]*(n)

d=list(range(0,n))
d_s=list(range(0,n))

o=0
fu_ci=[0]*(n_i)
fu_i=0
fgini_coeff=[]
fgini_coeff_ten=[]
fgini_coeff_twfive=[]
fgini_coeff_fifty=[]
fgini_coeff_sevfive=[]

ten_per_d=0
twfive_per_d=0
fifty_per_d=0
sevfive_per_d=0
ol=0
wt_avg=0
eff_avg=0
fol=582
up=[fol]*(n)
prev_up=[fol]*(n)
loc_cou=0
loc=[]
loc1=random.sample(range(0, 540), n)
for i in loc1:
  loc.append(map1[i])

for cd in range(1,n_i):

   target_G = np.load('C:/Users/Dell/Downloads/Route recomm env predicted data/NY2km1folder/Ny_data_opt_veh_cou/24/' +str(int(fol)) + '/target_G.npy', allow_pickle=True).tolist()

   reqs_matrix=np.add(reqs_matrix,target_G)
   t_s=map_r[fol]
   t_s=t_s+1

   fol=map_f[t_s]

for cd in range(1,n_i):
 o=0
 for ak in prev_up:
  target_G = np.load('C:\\Users\\Dell\\Downloads\\Route recomm env predicted data\\NY2km1folder\\Ny_data_opt_veh_cou\\24\\' +str(ak) + '/target_G.npy', allow_pickle=True).tolist()
  res_G = np.load('C:\\Users\\Dell\\Downloads\\Route recomm env predicted data\\NY2km1folder\\Ny_data_opt_veh_cou\\24\\' +str(ak) + '/res_G.npy', allow_pickle=True).tolist()
  target_G = np.asarray(target_G, dtype=np.float32)
  target_G = target_G.reshape(576, 576)

  np.save('C:\\Users\\Dell\\Downloads\\Route recomm env predicted data\\NY2km1folder\\Ny_data_opt_veh_cou\\25\\' +str(ak) + '/res_G.npy', res_G)
  np.save('C:\\Users\\Dell\\Downloads\\Route recomm env predicted data\\NY2km1folder\\Ny_data_opt_veh_cou\\25\\' +str(ak) + '/target_G.npy', target_G)

 zk=[]
 r_a=1
 driver_i=0

 for hm in d:
  capacity=2

  if driver_i==0:
   target_G = np.load('C:\\Users\\Dell\\Downloads\\Route recomm env predicted data\\NY2km1folder\\Ny_data_opt_veh_cou\\24\\' +str(up[hm]) + '/target_G.npy', allow_pickle=True).tolist()
   res_G = np.load('C:\\Users\\Dell\\Downloads\\Route recomm env predicted data\\NY2km1folder\\Ny_data_opt_veh_cou\\24\\' +str(up[hm]) + '/res_G.npy', allow_pickle=True).tolist()
   target_G = np.asarray(target_G, dtype=np.float32)
   target_G = target_G.reshape(576, 576)

   res_G = np.asarray(res_G, dtype=np.float32)
   res_G = res_G.reshape(576, 576)
   res_D = np.array(res_D)

   res_D = res_D.ravel()

  else:
   target_G = np.load('C:\\Users\\Dell\\Downloads\\Route recomm env predicted data\\NY2km1folder\\Ny_data_opt_veh_cou\\25\\' +str(up[hm]) + '/target_G.npy', allow_pickle=True).tolist()
   res_G = np.load('C:\\Users\\Dell\\Downloads\\Route recomm env predicted data\\NY2km1folder\\Ny_data_opt_veh_cou\\25\\' +str(up[hm]) + '/res_G.npy', allow_pickle=True).tolist()
   target_G = np.asarray(target_G, dtype=np.float32)
   target_G = target_G.reshape(576, 576)

   res_G = np.asarray(res_G, dtype=np.float32)
   res_G = res_G.reshape(576, 576)
   res_D = np.array(res_D)

   res_D = res_D.ravel()

  fwait_t=[]

  elements, loc_sparsity_arr, x, y, xind, yind, loc_arr = loc_sparsity(reqs_matrix[0])

  src=loc[o]
  path1=[]
  path2=[]
  path12=[]
  path=[]

  if driver_i==(n-1):
   eff_avg=sum(fu_ci)/cd
   wt_avg=sum(fwt)/(n_i*cd+hm)

  if cd!=0:

    src1=src

    dg_a, arr, r, c, dest, capacity, src, path, gr = DAG(res_G, src, capacity, gr)
    k = 0
    dag = np.zeros((abs(r) + 1, abs(c) + 1))
    for i in range(0, abs(r) + 1):
      for j in range(0, abs(c) + 1):
          dag[i][j] = map[arr[k]]
          k = k + 1

    dag = dag.astype(int)
    req_t = np.zeros((abs(r) + 1, abs(c) + 1))
    vertices = np.zeros((abs(r) + 1, abs(c) + 1))
    distances = np.zeros((abs(r) + 1, abs(c) + 1))
    vertices, req_t, distances, capacity = dp(r, c, src, dag, vertices, req_t, distances, capacity)

    print("src",src)
    src1=src=map1[src]
    path1=SP(src1,src)

    for i in path1:
      path2.append(map[i])

    path = []
    in_l = 0
    count = 6
    count1 = 6
    src = loc[hm]

    path.append(src)
    path2 = []
    if type(src) is tuple:
      src = map[src]

    while np.sum(target_G, axis=1)[src] == 0 and count > 0:
      dest = cal_max(src)

      print("dest", dest)

      path.append(dest)
      src = dest
      count = count - 1
    if count == 0:
      print("path in count", path)
    else:
      for i in range(0, 576):

          if res_D[i] > 0:

              path.append(i)
              print("path before", path)
              p1 = path
              path = checksp(path[-2:])
              path = list(dict.fromkeys(path))

              print("path after", path)
              for av in p1:
                  path2.append(av)
              for av in path:
                  path2.append(av)

              src = path2[-1]
              if len(path) > count1:

                  break
    print("path is", path)
    if type(dest) is not tuple:
        dest=map1[dest]

    xz = []
    for am in path:
       if am != '/':
          xz.append(am)
    print("xz", xz)
    res_list=xz
    for i in res_list:
      if type(i) is tuple:
          res_list[0] = map[i]
    for ab in res_list:
      if area_sparse[map1[ab]] == True:
          sparse[hm] = sparse[hm] + 1
      all[hm] = all[hm] + 1

    driver_i=driver_i+1
    time_skip=time_gr(loc[o],dest)
    time_skip=math.ceil(time_skip/15)
    prev_up[hm]=int(up[hm])
    up[hm]=int(map_r[up[hm]])
    up[hm]=int(up[hm]+time_skip)
    up[hm]=int(map_f[up[hm]])
    loc[o]=dest
    o=o+1

  res_list=xz
  target_G1 = target_G.copy()
  req_path=req_in_path(res_list)

  com_reqs=compatible_reqs(res_list,req_path)

  taken_reqs_matrix=loc_sparsity_check(res_list,com_reqs,taken_reqs_matrix,target_G1)

  path=[]
  if com_reqs:

      fwait_t.append(WT(res_list,com_reqs))
      s_m=0
      fare[hm]=fare_cal(com_reqs)
      for i in fwait_t:
        s_m=s_m+i[0]
      avg1=s_m/len(fwait_t)

      fwt[ol]=avg1
      ol=ol+1
      print("waiting time",fwait_t)
      fr[hm]=fr[hm]+fare_cal(com_reqs)

      if  type(dest) is tuple:
       distan[hm]=distan[hm]+dist_of_path(path,distances,dag,map[dest])
      else:
          distan[hm] = distan[hm] + dist_of_path(path, distances, dag, dest)

      try:
        tim=dist_of_path(path,distances,dag,map[dest])/3
        fu_c = (fare_cal(com_reqs) - dist_of_path(path, distances, dag, map[dest]) * (
                    0.202 / 1.55))

      except:
          tim=2
          fu_c=20

      fu_ci[cd]=fu_ci[cd]+fu_c
      try:
       futility[hm]=((r_a-1)*futility[hm] + (fu_c/tim))/r_a
      except:
          ap=1

  capacity=2
  print("com_reqs",com_reqs)
  for vc in com_reqs:
   print("vc",vc)
   o_x,o_y=vc
   target_G[o_x][o_y]=0
  np.save('C:\\Users\\Dell\\Downloads\\Route recomm env predicted data\\NY2km1folder\\Ny_data_opt_veh_cou\\25\\' +str(up[hm]) + '/res_G.npy', res_G)
  np.save('C:\\Users\\Dell\\Downloads\\Route recomm env predicted data\\NY2km1folder\\Ny_data_opt_veh_cou\\25\\' +str(up[hm]) + '/target_G.npy', target_G)
  print("saved")

 r_a=r_a+1
 s = np.array(futility)
 sort_index = np.argsort(s)
 d_s=list(sort_index)
 print("utility before",futility)
 futility=normalize(futility)

 fgini_coeff.append(gini(np.array(futility)))
 futility1=np.sort(futility)
 ten_per_d=math.floor(0.1*n)
 fgini_coeff_ten.append(gini(np.array(futility1[0:ten_per_d])))

 twfive_per_d=math.floor(0.25*n)
 fgini_coeff_twfive.append(gini(np.array(futility1[0:twfive_per_d])))

 fifty_per_d=math.floor(0.5*n)
 fgini_coeff_fifty.append(gini(np.array(futility1[0:fifty_per_d])))

 sevfive_per_d=math.floor(0.75*n)
 fgini_coeff_sevfive.append(gini(np.array(futility1[0:sevfive_per_d])))

 elements,loc_sparsity_arr,x,y,xind,yind,loc_arr=loc_sparsity(reqs_matrix[0])

print("sparse,all",sparse,all)
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
       la.append(reqs_matrix[0][i][j])

kl_rider(loc_sparsity_arr,la,np.sum(taken_reqs_matrix))

print("utility per hour",sum(futility)/len(futility))
print("platforms utility",sum(fu_ci)/len(fu_ci))
print("wt",sum(fwt)/len(fwt))