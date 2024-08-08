import numpy as np
from scipy.stats import chisquare

length_grid = 577
rows_grid = 24  # sqrt(length_grid-1)
cols_grid = 24
grid = np.arange(1, length_grid, 1, dtype=int)
grid = grid.reshape((cols_grid, rows_grid))

res_G = np.load(
    'C:\\Users\\Dell\\Downloads\\Route recomm env predicted data\\NY2km1folder\\Ny_data_opt_veh_cou\\24\\0\\res_G.npy',
    allow_pickle=True).tolist()
target_G = np.load(
    'C:\\Users\\Dell\\Downloads\\Route recomm env predicted data\\NY2km1folder\\Ny_data_opt_veh_cou\\24\\0\\target_G.npy',
    allow_pickle=True).tolist()
res_D = np.load(
    'C:\\Users\\Dell\\Downloads\\Route recomm env predicted data\\NY2km1folder\\Ny_data_opt_veh_cou\\24\\0\\res_D.npy',
    allow_pickle=True).tolist()
target_D = np.load(
    'C:\\Users\\Dell\\Downloads\\Route recomm env predicted data\\NY2km1folder\\Ny_data_opt_veh_cou\\24\\0\\target_D.npy',
    allow_pickle=True).tolist()


time_l = np.load(
    'C:\\Users\\Dell\\Downloads\\Route recomm env predicted data\\NY2km1folder\\Ny_data_opt_veh_cou\\24\\time_l.npy')  # load
folder = np.load(
    'C:\\Users\\Dell\\Downloads\\Route recomm env predicted data\\NY2km1folder\\Ny_data_opt_veh_cou\\24\\folder.npy')  # load
time_l = list(time_l)
folder = list(folder)

import math

hour = [0] * len(time_l)
rounded_hour = [0] * len(time_l)
day = [0] * len(time_l)
km = 0
for i in time_l:
    hour[km] = math.ceil(i / 4)
    rounded_hour[km] = hour[km] % 24
    day[km] = math.ceil(hour[km] / 24)
    km = km + 1

res_G = np.asarray(res_G, dtype=np.float32)

res_G = res_G.reshape(576, 576)
# print(np.where(target_D==0))


target_G = np.asarray(target_G, dtype=np.float32)
target_G = target_G.reshape(576, 576)

map_f = zip(time_l, folder)
map_f = dict(map_f)  # maps (0,0) to 0

map_r = zip(folder, time_l)
map_r = dict(map_r)  # maps (0,0) to 0

# Mapping
relative = []
for i in range(0, cols_grid):
    for j in range(0, rows_grid):
        relative.append(tuple([i, j]))
positional = list(range(0, rows_grid * cols_grid + 1, 1))
map = zip(relative, positional)
map = dict(map)  # maps (0,0) to 0

map1 = zip(positional, relative)  # maps 0 to (0,0)   i.e maps grid cells to coordinate
map1 = dict(map1)

sav = res_G


# calculates shortest path
def SP(source, des):
    a, b = source
    c, d = des
    shortest_path = [source]
    row = abs(c - a)  # move these many rows down
    col = abs(d - b)
    if c != a:
        row_dir = int(row / (c - a))
    else:
        row_dir = 1
    if d != b:
        col_dir = int(col / (d - b))
    # find min(row,col)
    else:
        col_dir = 1
    if row <= col:
        eql_move = row
        # row=0   #we dont need to move any more rows now
    else:
        eql_move = col
    # col=0
    row = row - eql_move
    col = col - eql_move
    for i in range(1, eql_move + 1):  # adds (2,2),(3,3),...
        a = a + row_dir
        b = b + col_dir
        shortest_path.append(tuple([a, b]))

    for j in range(0, col):
        b = b + col_dir
        shortest_path.append(tuple([a, b]))
    for j in range(0, row):
        a = a + row_dir
        shortest_path.append(tuple([a, b]))

    return shortest_path


def len_sp(source, dest):
    a, b = source
    c, d = dest
    shortest_path = [source]
    row = abs(c - a)  # move these many rows down
    col = abs(d - b)
    if c != a:
        row_dir = int(row / (c - a))
    else:
        row_dir = 1
    if d != b:
        col_dir = int(col / (d - b))
    # find min(row,col)
    else:
        col_dir = 1
    if row <= col:
        eql_move = row
        # row=0   #we dont need to move any more rows now
    else:
        eql_move = col
    # col=0
    row = row - eql_move
    col = col - eql_move
    for i in range(1, eql_move + 1):  # adds (2,2),(3,3),...
        a = a + row_dir
        b = b + col_dir
        shortest_path.append(tuple([a, b]))

    for j in range(0, col):
        b = b + col_dir
        shortest_path.append(tuple([a, b]))
    for j in range(0, row):
        a = a + row_dir
        shortest_path.append(tuple([a, b]))

    length = len(shortest_path) - 1
    return length


import sys


def findMax(arr):
    max = arr[0]
    n = len(arr)  # number of rows

    # Traverse array elements from second
    # and compare every element with
    # current max
    for i in range(1, n):
        if arr[i] > max:
            max = arr[i]
    return max


import math


def dist(cpx, cpy, gr):
    dist = []
    cor = []
    for i, j in zip(gr[0], gr[1]):
        if i != j:  # dont loop over same grid cell
            x, y = map1[i]
            dist.append(tuple([math.sqrt((cpx - x) ** 2 + (cpy - y) ** 2)]))
            cor.append(tuple([x, y]))

    d_m = min(dist)
    index = dist.index(d_m)
    val = cor[index]

    # c,v=val
    # m=c-r
    return val
    # n=v-t


n = 1000
loc_x = np.random.randint(0, high=23, size=n, dtype=int)
loc_y = np.random.randint(0, high=23, size=n, dtype=int)
loc = list(zip(loc_x, loc_y))  # np.array(zip(loc_x,loc_y))
# print("loc",loc)
# access loc[0]

capacity = 2
gr = np.where(res_G > 0)


# print("capacity is",capacity)
def DAG(res_G, source,
        capacity):  # will take as input full request(predicted) graph and generate DAG for current source
    # source is of form row index col index like (2,7)
    path = []
    arr = []
    dest_arr = []
    br = 0  # variable to break ou of loop
    # dag_arr=np.arange(1, 31*1, 1, dtype=int)
    # dag_arr=dag_arr.reshape((31,1))
    dag_arr = -100 * np.ones((576, 576))
    # dag_arr=[]
    # dag_arr=np.fullitialized to - inf

    sx, sy = source  # x and y coordinates of source
    src = map[source]
    tru = 0
    # for i in res_G[src]:
    #  if sum(np.array(res_G[src]))==i:
    #   tru=1
    if sum(np.array(res_G[
                        src])) == 0:  # or tru==1: #if there are no requests from current point to dest relocate drievr through shortest path to demand aware location
        val = dist(sx, sy, gr)
        # val conatins src and dest
        # val_s,val_d=val
        # move from shortest path from src to val and update src
        # v_s=map1[val_s]
        # print("val",v_s)
        path = SP(source, val)
        src = map[val]
        sx, sy = val  # x and y coordinates of source
    # print("capacity is,src is",capacity,src)
    dest = 0
    res_G[src][src] = 0  # cant loop over src
    dest = findMax(np.array(res_G[src]))

    # print("dest",dest)
    dest = np.where(res_G[src] == dest)[0][0]
    # print("capacity is",capacity)
    tempor2 = min(capacity, res_G[src][dest])
    res_G[src][dest] = res_G[src][dest] - tempor2

    capacity = capacity - tempor2
    # print("capacity after",capacity)
    destination = map1[dest]
    # print("new src is",sx,sy)
    # print("dest is",destination)
    dx, dy = destination
    # create DAG between source and destination
    r = sx - dx
    c = sy - dy

    for i in range(0, abs(r) + 1):
        for j in range(0, abs(c) + 1):
            if r != 0:
                dx = int(r / abs(r))  # direction to move
            else:
                dx = 0
            if c != 0:
                dy = int(c / abs(c))  # direction to move
            else:
                dy = 0
            # dy=c/abs(c) #direction to move
            # print("sx,sy,dx,dy,i,j",sx,sy,dx,dy,i,j)
            arr.append(tuple([int(sx - dx * i), int(sy - dy * j)]))

    # print("arr",arr)
    for f in arr:
        mp = map[f]
        #    print("f",mp)
        x, y = f
        #     print("x,y",x,y)
        if (x - dx, y) in arr:
            #     print("x-dx")
            dag_arr[map[(x, y)]][map[(x - dx, y)]] = res_G[map[(x, y)]][map[(x - dx, y)]]
        if (x, y - dy) in arr:
            dag_arr[map[(x, y)]][map[(x, y - dy)]] = res_G[map[(x, y)]][map[(x, y - dy)]]
        #      print("y-dy")
        if (x - dx, y - dy) in arr:
            dag_arr[map[(x, y)]][map[(x - dx, y - dy)]] = res_G[map[(x, y)]][map[(x - dx, y - dy)]]

    return dag_arr, arr, r, c, destination, capacity, src, path


# find shortest path from all vertices to destination
alpha = 1.7
sps = np.zeros(
    length_grid)  # generates requests matrix of length length_grid*length_grid. Each element of matrix is randomly generated between 0 and 6


def sp(dest):
    a, b = dest
    for i in range(0, length_grid - 1):  # could use optimized destination in place of length_grid
        # for j in range(0,length_grid):  #see a+1 or a
        sps[i] = len_sp(map1[i], dest)
    return sps



import math


def dp(r, c, src, dag, vertices, req_t, distances, capacity):
    # for j in range(0,abs(r)+1):
    # for i in dag[j]:
    req = []
    reqs = []
    for i in range(0, abs(r) + 1):
        for j in range(0, abs(c) + 1):
            req = []
            reqs = []  # temporary variables
            if i == 0 and j == 0:
                req_t[0][0] = 0
                vertices[0][0] = 0
                distances[0][0] = 0
            else:
                # print(dag[i][j])
                # i-1,j i,j-1 i-1,j-1
                #    print("sps[dag[i][j]]",sps[dag[i][j]])
                #     print("distances[dag[i][j-1]]",distances[dag[i][j-1]])
                if i - 1 >= 0:
                    #    print("i-1")
                    x = 0
                    x = alpha * sps[src] - sps[dag[i][j]] - distances[i - 1][j]  # [dag[i-1][j]]
                    if x >= 1:
                        #        req_t[i][j]=res_G[dag[i-1][j]][dag[i][j]] + req_t[i-1][j]
                        req.append(dag[i - 1][j])
                        reqs.append(res_G[dag[i - 1][j]][dag[i][j]] + req_t[i - 1][j])  # append req_t here

                if j - 1 >= 0:
                    #      print("j-1")
                    x = 0
                    #      req_t[i][j]=res_G[dag[i][j-1]][dag[i][j]] + req_t[i][j-1]
                    x = alpha * sps[src] - sps[dag[i][j]] - distances[i][j - 1]  # [dag[i][j-1]]
                    if x >= 1:
                        req.append(dag[i][j - 1])
                        reqs.append(res_G[dag[i][j - 1]][dag[i][j]] + req_t[i][j - 1])
                if i - 1 >= 0 and j - 1 >= 0:
                    x = 0
                    #     print("i-1,j-1")
                    # req_t[i][j]=res_G[dag[i-1][j]][dag[i][j]] + req_t[i-1][j-1]
                    x = alpha * sps[src] - sps[dag[i][j]] - distances[i - 1][j - 1]  # [dag[i-1][j-1]]
                    if x >= 1:
                        req.append(dag[i - 1][j - 1])
                        reqs.append(res_G[dag[i - 1][j - 1]][dag[i][j]] + req_t[i - 1][j - 1])
                if req:
                    mx = max(reqs)
                    ind = reqs.index(mx)
                    el = req[ind]
                    # print("max req,element",mx,el)
                    vertices[i][j] = el
                    ###################subtract hereeee
                    #    print("sub reqs")
                    #    print("req_g before",res_G[el][dag[i][j]])
                    tempor = min(capacity, res_G[el][dag[i][j]])
                    res_G[el][dag[i][j]] = res_G[el][dag[i][j]] - tempor
                    capacity = capacity - tempor
                    #  print("req_g after",res_G[el][dag[i][j]])
                    # print("caapcity after",capacity)
                    # we assume capacity of vehicle is 2 and 1 is already in vehicle so we subtract 1 as we can take 1 request only
                    #     a=np.where(dag==el)[0][0]
                    # a,b=dg_a.index(el)  #see how to retrun index of 2d array
                    z = list(zip(*np.where(dag == el)))
                    a = z[0][0]
                    b = z[0][1]
                    #  print("a",a)
                    req_t[i][j] = mx
                    distances[i][j] = distances[a][b] + 1
                else:  # make this vertex unreachable
                    req_t[i][j] = -100
                    distances[i][j] = 100
                    vertices[i][j] = -100
    return vertices, req_t, distances, capacity


def ret_path(vertices, src):
    dpath = [map[dest]]
    # print("dag",dag)
    # print("dest",dest)
    x = list(zip(*np.where(dag == map[dest])))
    # print("x is",x)
    a = int(vertices[x[0][0], x[0][1]])
    # print("map[src]",map[src])
    # dpath.append(a)
    while a != src:
        dpath.append(a)  # adds destination vertex
        # print("a",a)
        #  print("a",a)
        x = list(zip(*np.where(dag == a)))
        # print("x is",x)
        a = int(vertices[x[0][0], x[0][1]])
        # a=int(vertices[a])
    # print("dpath",dpath)
    return dpath


alpha = 1.7


def req_in_path(path):  # check detour ratiio here>>>>
    # use actual req set and see how many request does current path cover
    # we need to know requests origin and destination
    rq_p = []
    rq_comp = []
    # print("path in req_path",path)
    # print("len(path)",len(path))
    for i in range(0, len(path)):
        for j in range(i, len(path)):
            # print("path[i],path[j]",path[i],path[j])
            # print("reqs",res_G[path[i]][path[j]])
            if target_G[path[i]][path[j]]:
                rq_p.append(tuple([path[i], path[j]]))
    #  print("req_path in func",rq_p)
    # print("reqs",rq_p)
    for i in rq_p:
        a, b = i
        if a == b:
            rq_comp.append(tuple([a, b]))

        if a != b:
            if ((path.index(b) - path.index(a)) / len_sp(map1[a], map1[b])) <= alpha:#random.uniform(1, 2):
                rq_comp.append(tuple([a, b]))
    # print("comp reqs",rq_comp)
    return rq_comp


def compatible_reqs(path, req_path):
    capacity = 2
    edges = {}
    taken_reqs = []
    #  print("path,req_path",path,req_path)
    # for i in req_path:
    # x,y=i
    # if x==y:
    # taken_reqs.append(i)

    for first, second in zip(path, path[1:]):
        #  print("fitrst,second",first,second)
        edges[(first, second)] = 0  # initially all edges have 0 requests
    br = 0
    temp = []
    temp2 = []
    reqs_s = []
    cv = []
    # print("edges",edges)
    for c in req_path:
        #  print("c is",c[1],c[0])
        #    print("math.sqrt((c[1]-c[0])**2)",math.sqrt((c[1]-c[0])**2))
        temp.append(math.sqrt((c[1] - c[0]) ** 2))
        cv.append(math.sqrt((c[1] - c[0]) ** 2))
    # temp2.append(c)
    temp2 = temp
    temp2.sort(reverse=True)
    # print("temp",temp)
    # print("temp2",temp2)
    # print("cv",cv)
    arr = np.argsort(np.array(temp))
    brr = list(arr[::-1])
    if brr:
        for k in brr:
            #      ind=cv.index(k)
            reqs_s.append(req_path[k])
    # print("sorted reqs",reqs_s)
    for i in reqs_s:
        br = 0
        a, b = i
        ind1 = path.index(a)
        ind2 = path.index(b)
        # print("ind1,ind2",ind1,ind2)
        #  print("path",path)
        for first, second in zip(path[ind1:ind2 + 1], path[ind1 + 1:ind2 + 1]):
            edges[(first, second)] = edges[(first, second)] + target_G[path[ind1]][path[ind2]]
        for first, second in zip(path[ind1:ind2 + 1], path[ind1 + 1:ind2 + 1]):

            if edges[(first, second)] > capacity:
                br = 1
                for first, second in zip(path[ind1:ind2 + 1], path[ind1 + 1:ind2 + 1]):
                    edges[(first, second)] = edges[(first, second)] - target_G[path[ind1]][path[ind2]]

        if br == 0:  # req can be taken
            taken_reqs.append(i)
            m, n = i

    #######################################helloaqsa  target_G[m][n]=0  #as all reqs are taken #target_G[m][n]-capacity
    # for first, second in zip(path[ind1:ind2+1], path[ind1+1:ind2+1]):
    #  edges[(first,second)]=edges[(first,second)]-target_G[path[ind1]][path[ind2]]

    #  print("taken_reqs",taken_reqs)
    return taken_reqs


def compatible_reqs_sub(path, req_path):
    capacity = 2
    edges = {}
    taken_reqs = []
    #  print("path,req_path",path,req_path)
    # for i in req_path:
    # x,y=i
    # if x==y:
    # taken_reqs.append(i)

    for first, second in zip(path, path[1:]):
        #  print("fitrst,second",first,second)
        edges[(first, second)] = 0  # initially all edges have 0 requests
    br = 0
    temp = []
    temp2 = []
    reqs_s = []
    cv = []
    # print("edges",edges)
    for c in req_path:
        #  print("c is",c[1],c[0])
        #    print("math.sqrt((c[1]-c[0])**2)",math.sqrt((c[1]-c[0])**2))
        temp.append(math.sqrt((c[1] - c[0]) ** 2))
        cv.append(math.sqrt((c[1] - c[0]) ** 2))
    # temp2.append(c)
    temp2 = temp
    temp2.sort(reverse=True)
    # print("temp",temp)
    # print("temp2",temp2)
    # print("cv",cv)
    arr = np.argsort(np.array(temp))
    brr = list(arr[::-1])
    if brr:
        for k in brr:
            #      ind=cv.index(k)
            reqs_s.append(req_path[k])
    # print("sorted reqs",reqs_s)
    for i in reqs_s:
        br = 0
        a, b = i
        ind1 = path.index(a)
        ind2 = path.index(b)
        # print("ind1,ind2",ind1,ind2)
        #  print("path",path)
        for first, second in zip(path[ind1:ind2 + 1], path[ind1 + 1:ind2 + 1]):
            edges[(first, second)] = edges[(first, second)] + target_G[path[ind1]][path[ind2]]
        for first, second in zip(path[ind1:ind2 + 1], path[ind1 + 1:ind2 + 1]):

            if edges[(first, second)] > capacity:
                br = 1
                for first, second in zip(path[ind1:ind2 + 1], path[ind1 + 1:ind2 + 1]):
                    edges[(first, second)] = edges[(first, second)] - target_G[path[ind1]][path[ind2]]

        if br == 0:  # req can be taken
            taken_reqs.append(i)
            m, n = i

            target_G[m][n] = 0  # as all reqs are taken #target_G[m][n]-capacity
        # for first, second in zip(path[ind1:ind2+1], path[ind1+1:ind2+1]):
        #  edges[(first,second)]=edges[(first,second)]-target_G[path[ind1]][path[ind2]]

    #  print("taken_reqs",taken_reqs)
    return taken_reqs


def fare_cal(taken_reqs):
    fare = []
    bf = 2.55  # base fare
    p_m = 0.35
    p_mile = 1.75
    far = 0
    fr = 0
    mf = 7
    # 1 litre=1.01 USD can cover 12.5km
    # 2.5km can be done in 0.2 litres which implies 0.202 $
    # 15 minutes for 2.5km
    # avg speed=3mph   in northnumber avenue #https://www.forbes.com/sites/carl5/22/uber-data-reveals-motoring-slower-than-walking-in-many-cities/?sh=5c35f71c16fb
    # 0.129 petrol per mile as 2.5km is 1.55 miles
    # .129*1.01 is price of petrol for .129 litres
    if len(taken_reqs) > 1:
        for i in taken_reqs:
            #   print("map1[i[1]]",i[1])
            #   print("map1[i[0]]",i[0])
            dist = (len_sp(map1[i[0]], map1[i[1]]) + 1) * 1.24  # +1.55  #as we have done +1 so it counts the grid also
            time = dist / 0.05  # 3 miles per hour is 0.05 miles per minute
            fr = bf + p_m * time + p_mile * dist
            if fr < mf:
                far = mf
            else:
                far = fr
            fare.append(0.8 * far)
    else:
        # bf=3#2.55 #base fare ######################CHANGE fare
        # p_m=0.5#0.35
        # p_mile=2#21.75
        # far=0
        # fr=0
        # mf=9#7
        #   for i in taken_reqs:
        #   print("map1[taken_reqs[0]]",taken_reqs[0][0])
        # print("map1[taken_reqs[1]]",taken_reqs[0][1])
        dist = (len_sp(map1[taken_reqs[0][0]], map1[taken_reqs[0][1]]) + 1) * 1.24  # +1.55
        time = dist / 0.05
        fr = bf + p_m * time + p_mile * dist
        if fr < mf:
            far = mf
        else:
            far = fr
        fare.append(far)
        # math.sqrt((i[1]-i[0])**2)*2.5 +2.5
        # fare=fare  #
    # print("fare is",sum(fare))
    return sum(fare)


def lorenzcurve(X):
    X_lorenz = X.cumsum() / X.sum()
    # print("x_lorenz",X_lorenz)
    X_lorenz = np.insert(X_lorenz, 0, 0)
    X_lorenz[0], X_lorenz[-1]
    # print("x_lorenz", X_lorenz)
    return X_lorenz


# print("X_lorenz[0], X_lorenz[-1]",X_lorenz[0], X_lorenz[-1])


def dist_of_path(path, distances, dag, dest):
    a = list(zip(*np.where(dag == dest)))
    # print("a in dist_of_path",a)
    #############check if only one grid cell is there

    if path:
        # if distances[a[0][0]][a[0][1]]:
        return (distances[a[0][0]][a[0][1]] + len_sp(path[0],
                                                     path[len(path) - 1])) * 1.24 + 1.24  # *1.75  #2km=1.24 miles
    # https://www.globalpetrolprices.com/USA/New_York_City/gasoline_prices/
    # https://www.google.com/search?q=how+many+litres+of+petrol+for+2.5+km&rlz=1C1CHBF_en__972__972&oq=how+many+litres+of+petrol+for+2.5+km&aqs=chrome..69i57j33i160i579.13871j0j7&sourceid=chrome&ie=UTF-8
    else:
        return (distances[a[0][0]][a[0][1]]) * 1.24 + 1.24  # *1.75  #each grid cell is 2.5 km


def gini(arr):
    ## first sort
    sorted_arr = arr.copy()
    sorted_arr.sort()
    n = arr.size
    coef_ = 2. / n
    const_ = (n + 1.) / n
    weighted_sum = sum([(i + 1) * yi for i, yi in enumerate(sorted_arr)])
    return coef_ * weighted_sum / (sorted_arr.sum()) - const_


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
        # print("requests_grid",requests_grid)
        # print("i,res_G",i,type(res_G[0]))
        # print(sum(res_G[i]))
        requests_grid.append(sum(res_G[i]))
        target_grid.append(sum(target_G[i]))
    # print(requests_grid)
    # print(target_grid)
    area_s = zip(relative, target_grid)
    area_s = dict(area_s)
    # print(area_s)
    for i in range(0, 24):
        for j in range(0, 24):
            reqs_total = 0
            for a in range(0, k):
                for b in range(0, k):
                    # print(i+a,j+b)
                    if area_s.get((i + a, j + b)) != None:
                        reqs_total = reqs_total + area_s[(i + a, j + b)]

            for a in range(1, k):
                for b in range(0, k):
                    # print(i-a,j+b)
                    if area_s.get((i - a, j + b)) != None:
                        reqs_total = reqs_total + area_s[(i - a, j + b)]
            for a in range(0, k):
                for b in range(1, k):
                    # print(i+a,j-b)
                    if area_s.get((i + a, j - b)) != None:
                        reqs_total = reqs_total + area_s[(i + a, j - b)]
            for a in range(1, k):
                for b in range(1, k):
                    # print(i-a,j-b)
                    if area_s.get((i - a, j - b)) != None:
                        reqs_total = reqs_total + area_s[(i - a, j - b)]
            # print(reqs_total)
            if reqs_total <= 0:
                area_sparse[(i, j)] = True
            else:
                area_sparse[(i, j)] = False
    #  print(area_sparse)
    return area_sparse, area_s


area_sparse, area_s = categorize_area()  # area is considered as sparse if its 3 hop neighbors dont contain any request


# it is used for making non-myopic decision


def calc_rel(location, hops):
    i, j = location
    area_r = []
    req_area = []
    # print("location",location)
    for a in range(0, hops):
        for b in range(0, hops):
            #   print(i+a,j+b)
            if area_s.get((i + a, j + b)) != None:
                if area_s.get((i + a, j + b)) > 0:
                    area_r.append((i + a, j + b))
                    #    print(" area_r.append((i+a,j+b))",area_r)
                    req_area.append(area_s.get((i + a, j + b)))
    for a in range(1, hops):
        for b in range(0, hops):
            #    print(i-a,j+b)
            if area_s.get((i - a, j + b)) != None:
                if area_s.get((i - a, j + b)) > 0:
                    area_r.append((i - a, j + b))
                    #  print(" area_r.append((i-a,j+b))",area_r)

                    req_area.append(area_s.get((i - a, j + b)))  # req_area[(i-a,j+b)]= area_s.get((i-a,j+b))
    for a in range(0, hops):
        for b in range(1, hops):
            #   print(i+a,j-b)
            if area_s.get((i + a, j - b)) != None:
                if area_s.get((i + a, j - b)) > 0:
                    area_r.append((i + a, j - b))
                    #       print(" area_r.append((i+a,j-b))",area_r)

                    req_area.append(area_s.get((i + a, j - b)))  # req_area[(i+a,j-b)]= area_s.get((i+a,j-b))
    for a in range(1, hops):
        for b in range(1, hops):
            #   print(i-a,j-b)
            if area_s.get((i - a, j - b)) != None:
                if area_s.get((i - a, j - b)) > 0:
                    area_r.append((i - a, j - b))
                    #     print(" area_r.append((i+a,j-b))",area_r)

                    req_area.append(area_s.get((i - a, j - b)))  # req_area[(i-a,j-b)]= area_s.get((i-a,j-b))
    # print("area_r",area_r)
    # print("area_s",area_s)
    # print("req_area",req_area)

    return area_r, req_area


def relocate(location, hops):
    # exit=0
    in_l = 0
    area_r = []
    req_area = []

    area_r, req_area = calc_rel(location, hops)
    sorted_area = [0] * (10000)
    sorted_req_area = [0] * (10000)  # *len(req_area))
    #  print("reqs in hops",req_area)
    # print("area_r",area_r)
    while len(req_area) == 0:
        area_r, req_area = calc_rel(location, hops + 1)
    # print(np.array(req_area))
    indices = np.argsort(np.array(req_area))
    #   print("indices",indices)
    k = 0
    for i in indices[::-1]:
        print(area_r[i])
        #    print("k",k)
        #   print(sorted_area)
        #  print(sorted_area[k])
        sorted_area[k] = area_r[i]
        sorted_req_area[k] = req_area[i]
        k = k + 1
    # print("sorted reqs in hops",sorted_req_area)
    # print("sorted area_r",sorted_area)
    for i in sorted_area:
        if area_sparse[i] == False:
            max_v = i  # sorted_req_area[area]
            #    print("max value",max_v)
            in_l = in_l + 1
            return i
        elif in_l >= len(sorted_area):
            #   print("in elif",in_l,len(sorted_area))
            return sorted_area[0]


def calc_rel_prev(location, hops):
    i, j = location
    req_area = {}
    for a in range(0, hops):
        for b in range(0, hops):
            # print(i+a,j+b)
            if area_s.get((i + a, j + b)) != None:
                if area_s.get((i + a, j + b)) > 0:
                    req_area[(i + a, j + b)] = area_s.get((i + a, j + b))
    for a in range(1, hops):
        for b in range(0, hops):
            # print(i-a,j+b)
            if area_s.get((i - a, j + b)) != None:
                if area_s.get((i - a, j + b)) > 0:
                    req_area[(i - a, j + b)] = area_s.get((i - a, j + b))
    for a in range(0, hops):
        for b in range(1, hops):
            # print(i+a,j-b)
            if area_s.get((i + a, j - b)) != None:
                if area_s.get((i + a, j - b)) > 0:
                    req_area[(i + a, j - b)] = area_s.get((i + a, j - b))
    for a in range(1, hops):
        for b in range(1, hops):
            # print(i-a,j-b)
            if area_s.get((i - a, j - b)) != None:
                if area_s.get((i - a, j - b)) > 0:
                    req_area[(i - a, j - b)] = area_s.get((i - a, j - b))
    return req_area


def relocate_prev(location, hops):
    # exit=0
    req_area = calc_rel_prev(location, hops)
    #   print("reqs in hops",req_area)
    while not bool(req_area):
        req_area = calc_rel_prev(location, hops + 1)

    max_v = max(zip(req_area.values(), req_area.keys()))[1]

    return max_v


def order_ride(path, reqs):
    b = c


import random


def WT(path, reqs):
    wt = []
    # print("path",path)
    # path should include relocation also>>
    for i in reqs:
        s, d = i
        #  print("s",s)
        #    t_s=path.index(s)*77.5-random.randint(path.index(s)*58,
        t_s = (path.index(s) - random.randint(0,
                                              path.index(s))) * 1.24 + random.uniform(0,
                                                                                      1.24)  # 1.55/0.05)  #3/4 of 77 is 57.75

        wt.append(t_s / 0.05)  # -#path[0]
    # we assume driver is at index 0
    return wt


def cal_num(path, start, end):
    #  print("start,end",start,end)
    start_index = path.index(start)
    end_index = path.index(end)
    #  print("path,start,end",path,start,end)
    return abs(end_index - start_index)


# SHORTEST PATH FULL WITHOUT k
def metrics_cal(pathh):
    orders = 0
    ok = 0
    el = 0
    added_elem = []
    efficiency = 0
    lk = []
    ct = 0
    capacity = 2
    c = capacity
    j = 0
    pass_g = 0  # passengers per grid
    copied_path = np.zeros(len(pathh))
    copied_path = list(copied_path)

    sum_eff = 0
    for i in range(len(pathh)):
        if pathh[i] != '/':
            copied_path[j] = pathh[i]
            # print("path,cop_path",pathh,copied_path)
            j = j + 1
    # print("copied path",copied_path)
    l_c = len(copied_path)
    i_c = 1
    p_k = copied_path
    reversed_list = copied_path[::-1]
    count = 0
    for iz in p_k:
        for kl in range(i_c - 2, -1, -1):  # path_k[:count]:
            #   print("added elem in stop",added_elem)
            #  print("[p_k[kl],iz] in added_elem",[p_k[kl],iz],[p_k[kl],iz] in added_elem)
            if target_G[p_k[kl]][iz] and [p_k[kl], iz] in added_elem:
                capacity = min(c, capacity + target_G[p_k[kl]][iz])
        #   print("cap in stop",capacity)
        i_c = i_c + 1
        count = count + 1
        lk = reversed_list[:-count]
        #   added_elem=[]
        for a in lk:
            if capacity > 0:
                val = cal_num(copied_path, a, iz)
                # print("val,alpha",val,alpha)
                # print("len",len_sp(map1[iz],map1[a]))
                #      print("iz,a",iz,a)
                if val <= alpha * len_sp(map1[iz], map1[a]):  # sps[map1[iz]][map1[a]]:
                    #     print("iz,a",iz,a)
                    capacity = max(0, capacity - target_G[iz][a])
                    added_elem.append([iz, a])
                # print("added elem",added_elem)
        # capacity indicates remaining capacity
        # cap=c-capacity
        pass_g = pass_g + c - capacity
        # percentage of orders without ridesharing
        if c - 2 >= capacity:  # if there are atleast 2 passengers in the vehicle then  ridesharing is successful
            ok = ok + 1
        if c - 1 == capacity:
            orders = orders + 1

        # efficiency
        # print("c,capacity",c,capacity)
        efficiency = (c - capacity) / c
        # print("eff is",efficiency)
        sum_eff = sum_eff + efficiency
    pass_g = pass_g / l_c  # (len_back[-1]-b_s)
    sum_eff = sum_eff / l_c  # (len_back[-1]-b_s)
    #  print("sum_eff is",sum_eff)
    perc_ride_without = (orders * 100) / l_c  # (len_back[-1]-b_s)
    perc_ride_with = (ok * 100) / l_c  # (len_back[-1]-b_s)

    # print("percentage of orders without ridesharing are",perc_ride_without)
    # rint("percentage of orders with ridesharing are",perc_ride_with)
    # print("passengers per grid",pass_g)
    return pass_g, perc_ride_with, perc_ride_without


def time_gr(loc1, loc2):
    dist = len_sp(loc1, loc2)
    dist = dist * 1.24  # miles
    tim = dist / 3  # time in hours
    tim = tim * 60
    return math.ceil(tim)


def loc_sparsity_check(res_list, com_reqs, taken_reqs_matrix, target_G1):
    for i, j in com_reqs:
        # print("i, j in comp reqs",i,j,target_G1[i][j])
        taken_reqs_matrix[i][j] = taken_reqs_matrix[i][j] + target_G1[i][j]  ###1?
    return taken_reqs_matrix


def find_elements_less_than_one_third_sorted(arr):
    # Flatten the 2D array into a 1D array
    # print("len(arr)",len(arr))
    flat_arr = arr.copy()  # [num for sublist in arr for num in sublist if num > 0]

    if len(flat_arr) == 0:
        return []  # No positive integers found

    flat_arr.sort()  # Sort the array

    index = len(flat_arr) // 3  # Calculate the 1/3rd position index

    if index >= len(flat_arr):
        return []  # Index out of range

    one_third_element = flat_arr[index]  # Get the 1/3rd smallest positive integer

    # Return all elements less than the 1/3rd smallest positive integer
    less_than_one_third = [num for num in flat_arr if num < one_third_element]
    # print("less",less_than_one_third)
    # print("flat_arr[:1000]",flat_arr[:1000])
    return flat_arr[:1000]  # less_than_one_third


def find_k_elements(la, loc_sparsity_arr, x, y, k):
    #   print("x,y",x,y)
    xarr = []
    yarr = []
    da = la.copy()
    # print("len(da)",len(da))
    da.sort()
    #  print("da",da)
    da = da[:k]
    az = 0

    a = la.copy()
    b = da.copy()
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
    # print("indices", indices)
    for i in indices:
        #   print("i",i,len(x),len(indices))
        xarr.append(x[i])
        yarr.append(y[i])
    #  print("xarr,yarr", xarr, yarr)
    return xarr, yarr


def loc_sparsity(reqs_matrix):
    # print(reqs_matrix[0])
    loc_sparsity_arr = reqs_matrix.copy()
    # print("taken_reqs>0",np.where(taken_reqs_matrix>0))

    # print("reqs_matrix>0", np.where(reqs_matrix > 0))
    for i in range(0, 576):
        for j in range(0, 576):
            if reqs_matrix[i][j] != 0:
                loc_sparsity_arr[i][j] = taken_reqs_matrix[i][j] / reqs_matrix[i][j]

    # print("loc_sp_ar>0", np.where(loc_sparsity_arr > 0))
    x = []
    y = []
    for i in range(0, 576):
        for j in range(0, 576):
            if reqs_matrix[i][j] > 0:
                x.append(i)
                y.append(j)
    # print("len(x",len(x),len(y))
    la = []
    lb = []
    for i, j in zip(x, y):
        la.append(loc_sparsity_arr[i][j])
        lb.append(reqs_matrix[i][j])

    elements = find_elements_less_than_one_third_sorted(np.array(la))
    k1 = 1000

    # print("elemets less than 1/3rd", elements)
    xind, yind = find_k_elements(la, loc_sparsity_arr, x, y, k1)
    # print("xind,yind",xind,yind)
    return elements, la, x, y, xind, yind, lb


import statistics

###2471 is len(da)
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


def sort_drivers(d, dense, all):
    # np.random.seed(0)
    # .... n = 30
    drivers = []
    init_loc = np.random.randint(0, high=n, dtype=int)
    # print("dense",dense)
    # print("all",all)
    loc_reqs = []
    for item1, item2 in zip(dense, all):
        if item2 != 0:
            loc_reqs.append(
                item1 / item2)  # np.fromfunction(np.vectorize(generate_fraction), (dense,all,n, 1), dtype=object)
        else:
            loc_reqs.append(0)
    # print("loc_reqs",loc_reqs)
    while loc_reqs[init_loc] == 0:
        init_loc = np.random.randint(0, high=n, dtype=int)
    prop_loc = init_loc
    count = 0
    # print("init loc",init_loc)
    # print("prop_loc",prop_loc)
    for i in range(0, 35):  # 100000
        if init_loc == prop_loc:
            drivers.append(init_loc)
            count = count + 1
        prop_loc = int(random.uniform(0, n))
        while loc_reqs[prop_loc] == 0:
            prop_loc = int(random.uniform(0, n))
        #  print("loc_reqs[init_loc]",loc_reqs[init_loc])
        # print("loc_reqs[prop_loc]",loc_reqs[prop_loc])
        if loc_reqs[init_loc] != 0:
            a_p = loc_reqs[prop_loc] / loc_reqs[init_loc]
        else:
            a_p = 0
        if a_p >= 1:
            init_loc = prop_loc

        else:

            r = random.random()
            if a_p >= r:
                init_loc = prop_loc

            else:
                init_loc = init_loc

    # print("matrix before",loc_reqs)

    matrix2 = loc_reqs
    # for i in range(1,n):
    # fraction = loc_reqs[i, 0]
    # modified_fraction = fraction.numerator/fraction.denominator
    # matrix2[i, 0] = modified_fraction

    #  print("matrix2 after",matrix2)

    #  print("drivers with duplicate",drivers)
    drivers_noduplicate = f7(drivers)  # [*set(drivers)]
    # print("drivers_reverse",len(drivers_noduplicate))
    # drivers_noduplicate=list(drivers_noduplicate)
    # print("drivers",drivers_noduplicate)
    if len(drivers_noduplicate) > 0:
        return drivers_noduplicate[::-1], loc_reqs
    else:
        return d, loc_reqs


def test_driver(driver_loc_arr, sparse):
    # driver_loc_arr.append(1-sum(driver_loc_arr))
    a1 = []
    b1 = []
    for i in driver_loc_arr:
        a1.append(i / sum(driver_loc_arr))
    # a1 = [i for i in a1 if i != 0]
    probabilities = np.array(a1)

    # Calculate the expected frequencies
    exp = [sum(sparse) / n] * (n)
    exp[-1] = sum(sparse) - sum(exp[:-1])
    # exp.append(0)
    for i in exp:
        b1.append(i / sum(exp))
    # b1 = [i for i in b1 if i != 0]
    expected_prob = np.array(b1)
    # print(probabilities)
    # print(exp)

    #  print("sum is",sum(probabilities))
    #    print(sum(expected_prob))
    # Perform the chi-square test
    test_statistic, p_value = chisquare(f_obs=probabilities, f_exp=expected_prob)

    # Print the test results
    print("Chi-square test statistic:", test_statistic)
    print("P-value:", p_value)


import numpy as np


def kl_divergence(p, q):
    """Calculate KL divergence between two discrete distributions."""
    epsilon = 1e-9  # 1e-9  # A small epsilon value to avoid division by zero
    q = np.where(q == 0, epsilon, q)  # Replace 0s in q with epsilon
    p = np.where(p == 0, epsilon, p)
    # print("p,q",p,q)
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))


# Example usage:
def driver_kl_diver(driver_loc_arr, sparse):
    a1 = []
    b1 = []
    print("driver_loc_arr", driver_loc_arr)
    for i in driver_loc_arr:
        a1.append(i / sum(driver_loc_arr))
    # a1 = [i for i in a1 if i != 0]
    probabilities = np.array(a1)

    # Calculate the expected frequencies
    exp = [n / sum(sparse)] * (n)
    # exp.append(0)
    for i in exp:
        b1.append(i / sum(exp))
    # b1 = [i for i in b1 if i != 0]
    expected_prob = np.array(b1)
    kl = kl_divergence(probabilities, expected_prob)
    print("KL Divergence driver:", kl)


def kl_rider(loc_sparsity_arr, reqs, num):  # reqs are total reqs in the area it doesnt take taken requests into account
    a1 = []
    b1 = []

    print(len(reqs))
    sum_of_elements = np.sum(reqs)
    #  print("rqes",reqs)
    # Normalize the array by dividing each element by the sum
    normalized_array = reqs / sum_of_elements
    #  print("na divided",normalized_array)
    normalized_array = normalized_array * num
    # print("na_mul",normalized_array)
    normalized_array = np.round(normalized_array)

    while np.sum(normalized_array) > num:
        jk = random.randint(0, len(normalized_array) - 1)
        # print("jk",jk)
        #   if normalized_array[jk]>1:
        normalized_array[jk] = max(normalized_array[jk] - 1, 1)
    # print("np.sum(normalized_array)2",np.sum(normalized_array))

    reqs_final = normalized_array / reqs
    # print("reqs_final",reqs_final)

    kl = kl_divergence(np.array(loc_sparsity_arr), np.array(reqs_final))
    print("KL Divergence rider:", kl)


import itertools

akv = 200


def nearby_drivers(loc):
    # n = math.ceil(576/5)  # Number of arrays which is num of vertices/5
    a = -1
    e = 6  # 2#3#3#5#4#5#8#6#4#2
    driver_ind = [[] for _ in range(n)]
    # print("loc",loc)
    cln = [[] for _ in range(n)]
    #    cln[a].append(loc[])
    loc_copy = loc.copy()
    for i in loc_copy:
        # cln.append(i)
        a = a + 1
        for j in loc_copy[loc_copy.index(i) + 1:]:
            if abs(j[0] - i[0]) <= e and abs(j[1] - i[1]) <= e:
                cln[a].append(j)
                driver_ind[a].append(loc_copy.index(j))

            # loc_copy.remove(j)
    print("locs", loc)
    print("coalitions", cln)
    print("driver_ind", driver_ind)
    # for i in cln:
    #    for j in i:
    #    driver_ind.append(loc.index(j))
    return driver_ind


reqs_matrix = np.zeros((576, 576), dtype=np.float32)  # res_G
taken_reqs_matrix = np.zeros((576, 576), dtype=np.float32)  # np.asarray(target_G, dtype=np.float32)

n = akv  # 10#30#5#30 #150#60
n_i = 3  # 3  # 32#50#20#100#10#100
fr = [0] * (n)
fr_s = [0] * (n)
distan = [0] * (n)
futility = [0] * (n)
futility_new = [0] * (n)
fwt = [0] * (n * n * n * n_i)

res_list_ar = [[] for _ in range(n + 1)]
dense = [0] * (n)
sparse = [0] * (n)
all = [0] * (n)
s_t = [0] * (n)
dstn = [(0, 0)] * (n)
d = list(range(0, n))  # driver indices
d_s = list(range(0, n))  # drivers sorted by their income
# 40#6 #num of iterations
o = 0
fu_ci = [0] * (n_i)
fu_i = 0

loc_x = np.random.randint(0, high=23, size=n, dtype=int)
loc_y = np.random.randint(0, high=23, size=n, dtype=int)
loc = list(zip(loc_x, loc_y))  # np.array(zip(loc_x,loc_y))
cln = nearby_drivers(loc)

ten_per_d = 0
twfive_per_d = 0
fifty_per_d = 0
sevfive_per_d = 0
ol = 0
cad = []
wt_avg = 0
eff_avg = 0
eff_avg = 0
fol = 0#1158  # 582#1158#0#1158#  1158
sparse_total = 0
up = [fol] * (n + 1)
prev_up = [fol] * (n + 1)
loc_cou = 0
fi = 0
inif = 1
fwaiting = []
iterations = 32
counting = []
for azm in range(0, iterations):
    for cd in range(1, n_i):  # see how 1000 drivers earn over 100 iterations
        o = 0

        for ak in prev_up:
            target_G = np.load(
                'C:\\Users\\Dell\\Downloads\\Route recomm env predicted data\\NY2km1folder\\Ny_data_opt_veh_cou\\24\\' + str(
                    ak) + '/target_G.npy', allow_pickle=True).tolist()
            res_G = np.load(
                'C:\\Users\\Dell\\Downloads\\Route recomm env predicted data\\NY2km1folder\\Ny_data_opt_veh_cou\\24\\' + str(
                    ak) + '/res_G.npy', allow_pickle=True).tolist()
            target_G = np.asarray(target_G, dtype=np.float32)
            target_G = target_G.reshape(576, 576)
            res_G = np.asarray(res_G, dtype=np.float32)
            res_G = res_G.reshape(576, 576)
            np.save(
                'C:\\Users\\Dell\\Downloads\\Route recomm env predicted data\\NY2km1folder\\Ny_data_opt_veh_cou\\28\\' + str(
                    ak) + '/res_G.npy', res_G)  # .detach().numpy())
            np.save(
                'C:\\Users\\Dell\\Downloads\\Route recomm env predicted data\\NY2km1folder\\Ny_data_opt_veh_cou\\28\\' + str(
                    ak) + '/target_G.npy', target_G)  # .detach().numpy())

        zk = []
        r_a = 1
        driver_i = 0
        for aq in loc:
            zk.append(map[aq])

        drivers = []
        if cd == 1:
            drivers = d
        else:
            #         drivers=[]
            drivers.append(random.randint(0, n - 1))
        for hm in drivers:
            print("drivers", drivers)
            capacity = 2
            f1 = 0
            fwait_t = []

            src = loc[hm]  # or you can keep loc[0] simply
            path1 = []
            path2 = []
            path12 = []
            path = []

            if cd > 1:
                driver_i = driver_i + 1

            if driver_i == 0:  # in first iteration every user calculates optimal route on initial graph
                target_G = np.load(
                    'C:\\Users\\Dell\\Downloads\\Route recomm env predicted data\\NY2km1folder\\Ny_data_opt_veh_cou\\24\\' + str(
                        up[hm]) + '/target_G.npy', allow_pickle=True).tolist()
                res_G = np.load(
                    'C:\\Users\\Dell\\Downloads\\Route recomm env predicted data\\NY2km1folder\\Ny_data_opt_veh_cou\\24\\' + str(
                        up[hm]) + '/res_G.npy', allow_pickle=True).tolist()
                target_G = np.asarray(target_G, dtype=np.float32)
                target_G = target_G.reshape(576, 576)
                res_G = np.asarray(res_G, dtype=np.float32)
                res_G = res_G.reshape(576, 576)

                dg_a, arr, r, c, dest, capacity, src, path = DAG(res_G, src, capacity)
                sps = sp(dest)
                dag = np.zeros((abs(r) + 1, abs(c) + 1))
                k = 0
                for i in range(0, abs(r) + 1):
                    for j in range(0, abs(c) + 1):
                        dag[i][j] = map[arr[k]]
                        k = k + 1
                # print("dag,src",dag,src)
                dag = dag.astype(int)
                req_t = np.zeros((abs(r) + 1, abs(c) + 1))
                vertices = np.zeros((abs(r) + 1, abs(c) + 1))
                distances = np.zeros((abs(r) + 1, abs(c) + 1))
                vertices, req_t, distances, capacity = dp(r, c, src, dag, vertices, req_t, distances, capacity)
                pat = ret_path(vertices, src)
                # print("path",pat)
                pat.append(src)
                for lko in path:
                    path12.append(map[lko])
                ab = itertools.chain(path2[:-1], path12[:-1], pat[::-1])
                res_list = list(ab)
                res_list = res_list[:5]
                dstn[hm] = map1[res_list[-1]]
                res_list_ar[hm] = res_list
                req_path = req_in_path(res_list)
                target_G1 = target_G.copy()
                com_reqs = compatible_reqs(res_list, req_path)
                if com_reqs:

                    fwait_t.append(WT(res_list, com_reqs))
                    fwaiting.append(WT(res_list, com_reqs))

                    s_m = 0
                    for i in fwait_t:
                        s_m = s_m + i[0]
                    avg1 = s_m / len(fwait_t)

                    fwt[ol] = avg1
                    ol = ol + 1

                    fr[hm] = fr[hm] + fare_cal(com_reqs)

                    distan[hm] = distan[hm] + dist_of_path(path, distances, dag, map[dest])

                    tim = dist_of_path(path, distances, dag,
                                       map[dest]) / 3  # 0.05   #dist/speed gives time  #time in hours
                    fu_c = (fare_cal(com_reqs) - dist_of_path(path, distances, dag, map[dest]) * (
                            0.1616 / 1.24))  # 1.55 miles can be done in 0.202$
                    # u_c is utilitarian utility

                    fu_ci[cd] = fu_ci[cd] + fu_c
                    futility[hm] = fu_c / tim  # ((r_a-1)*futility[hm] + (fu_c/tim))/r_a  #utility per hour

                capacity = 2



            else:
                inif = 0
                print("len(drivers)", len(drivers), drivers)
                if len(drivers) > 1:
                    driver = drivers[random.randint(0,
                                                    len(drivers) - 1)]  # random.randint(0,n-1)]  # random.randint(0,n)  #generate a driver randomly to update routes

                else:
                    driver = drivers[0]
                count = 1
                drivers.remove(driver)  # hm)
                print("res_list_ar[driver]", res_list_ar[driver])
                req_path = req_in_path(res_list_ar[driver])
                print("req_path", req_path)
                com_reqs = compatible_reqs_sub(res_list_ar[driver], req_path)
                for vc in com_reqs:  # the driver that broadcasted their requests are subtracted by others
                    print("vc", vc)
                    o_x, o_y = vc
                    target_G[o_x][o_y] = 0
                    res_G[o_x][o_y] = 0

                np.save(
                    'C:\\Users\\Dell\\Downloads\\Route recomm env predicted data\\NY2km1folder\\Ny_data_opt_veh_cou\\28\\' + str(
                        up[hm]) + '/res_G.npy', res_G)  # .detach().numpy())
                np.save(
                    'C:\\Users\\Dell\\Downloads\\Route recomm env predicted data\\NY2km1folder\\Ny_data_opt_veh_cou\\28\\' + str(
                        up[hm]) + '/target_G.npy', target_G)  # .detach().numpy())
                # drivers=[]  #see is it at correct position
                cln = nearby_drivers(loc)
                print("cln", cln)
                #      counter=[]
                for id in cln[driver]:  # range(0,30):#cln[driver]:
                    #   cr=cr+1

                    src = loc[id]
                    target_G = np.load(
                        'C:\\Users\\Dell\\Downloads\\Route recomm env predicted data\\NY2km1folder\\Ny_data_opt_veh_cou\\28\\' + str(
                            up[hm]) + '/target_G.npy', allow_pickle=True).tolist()
                    res_G = np.load(
                        'C:\\Users\\Dell\\Downloads\\Route recomm env predicted data\\NY2km1folder\\Ny_data_opt_veh_cou\\28\\' + str(
                            up[hm]) + '/res_G.npy', allow_pickle=True).tolist()
                    target_G = np.asarray(target_G, dtype=np.float32)
                    target_G = target_G.reshape(576, 576)
                    res_G = np.asarray(res_G, dtype=np.float32)
                    res_G = res_G.reshape(576, 576)

                    #                res_list_ar[id]
                    req_path = req_in_path(
                        res_list_ar[id])  # first recalculate the requests in previous path after broadcast by driver 1
                    target_G1 = target_G.copy()
                    com_reqs = compatible_reqs(res_list_ar[id], req_path)

                    if com_reqs:

                        fwait_t.append(WT(res_list_ar[id], com_reqs))
                        s_m = 0
                        for i in fwait_t:
                            s_m = s_m + i[0]
                        avg1 = s_m / len(fwait_t)

                        fwt[ol] = avg1
                        ol = ol + 1

                        #                   fr[id] = fr[id] + fare_cal(com_reqs)

                        #                    distan[id] = distan[id] + dist_of_path(path, distances, dag, map[dest])

                        tim = dist_of_path(path, distances, dag,
                                           map[dest]) / 3  # 0.05   #dist/speed gives time  #time in hours
                        fu_c = (fare_cal(com_reqs) - dist_of_path(path, distances, dag, map[dest]) * (
                                0.1616 / 1.24))  # 1.55 miles can be done in 0.202$
                        futility[id] = fu_c / tim
                    else:
                        futility[id] = 0


                    capacity = 2
                    ##now calculate utilities on modified graph
                    dg_a, arr, r, c, dest, capacity, src, path = DAG(res_G, src, capacity)
                    sps = sp(dest)
                    dag = np.zeros((abs(r) + 1, abs(c) + 1))
                    k = 0
                    for i in range(0, abs(r) + 1):
                        for j in range(0, abs(c) + 1):
                            dag[i][j] = map[arr[k]]
                            k = k + 1
                    # print("dag,src",dag,src)
                    dag = dag.astype(int)
                    req_t = np.zeros((abs(r) + 1, abs(c) + 1))
                    vertices = np.zeros((abs(r) + 1, abs(c) + 1))
                    distances = np.zeros((abs(r) + 1, abs(c) + 1))
                    vertices, req_t, distances, capacity = dp(r, c, src, dag, vertices, req_t, distances, capacity)
                    pat = ret_path(vertices, src)
                    # print("path",pat)
                    pat.append(src)
                    for lko in path:
                        path12.append(map[lko])
                    ab = itertools.chain(path2[:-1], path12[:-1], pat[::-1])
                    res_list = list(ab)
                    res_list = res_list[:5]
                    dstn[hm] = map1[res_list[-1]]
                    res_list_ar[hm] = res_list
                    req_path = req_in_path(res_list)
                    target_G1 = target_G.copy()
                    com_reqs = compatible_reqs(res_list, req_path)
                    if com_reqs:

                        fwait_t.append(WT(res_list, com_reqs))
                        s_m = 0
                        for i in fwait_t:
                            s_m = s_m + i[0]
                        avg1 = s_m / len(fwait_t)

                        fwt[ol] = avg1
                        ol = ol + 1

                        fr[id] = fr[id] + fare_cal(com_reqs)

                        distan[id] = distan[id] + dist_of_path(path, distances, dag, map[dest])

                        tim = dist_of_path(path, distances, dag,
                                           map[dest]) / 3  # 0.05   #dist/speed gives time  #time in hours
                        fu_c = (fare_cal(com_reqs) - dist_of_path(path, distances, dag, map[dest]) * (
                                0.1616 / 1.24))  # 1.55 miles can be done in 0.202$
                        # u_c is utilitarian utility

                        #                    fu_ci[cd] = fu_ci[cd] + fu_c
                        #  if inif == 1:
                        #     futility[hm] = fu_c / tim  # ((r_a-1)*futility[hm] + (fu_c/tim))/r_a  #utility per hour
                        # else:
                        futility_new[id] = fu_c / tim  # ((r_a-1)*futility[hm] + (fu_c/tim))/r_a  #utility per hour
                        if futility_new[id] > futility[id]:
                            futility[id] = futility_new[id]
                            drivers.append(id)  # update driver whose utility improved
                            drivers = set(drivers)
                            drivers = list(drivers)
                            count = count + 1
                    else:
                        print("no com reqs in new path")
                    counting.append(count)
                    capacity = 2


        r_a = r_a + 1
        print("futility", futility)
        s = np.array(futility)
        sort_index = np.argsort(s)
        d_s = list(sort_index)

        futility = normalize(futility)
        futility1 = np.sort(futility)
        for hm in d:
            time_skip = time_gr(loc[hm], dstn[hm])  # returns time to skip in minutes
            time_skip = math.ceil(time_skip / 15)  # since time is divided in 15 minute slots
            prev_up[hm] = int(up[hm])
            up[hm] = int(map_r[up[hm]])  # give time slot it will give folder name
            up[hm] = int(up[hm] + time_skip)
            up[hm] = int(map_f[up[hm]])
            loc[hm] = dstn[hm]  # will next time use their future destination
# print("loc_sparsity_arr",loc_sparsity_arr)

print("gini coefficient", gini(np.array(futility1)))
print("utility per hour", sum(futility) / len(futility))
print("platforms utility", sum(fu_ci) / len(fu_ci))
print("wt", sum(fwt) / len(fwt))
