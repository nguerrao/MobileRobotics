import numpy as np
import heapq

def heuristic(a, b):
    return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)

def redressRoad(route):
    data=[]
    for i in (range(0,len(route))):
        x = route[i][0]
        y = route[i][1]
        data.append((y,x))
    return data


def astar(obstaclesGrid, height, width, start, goal):
    start=(start[1],start[0])
    goal=(goal[1],goal[0])
    array=obstaclesGrid
    neighbors=[(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]
    close_set=set()
    came_from={}
    gscore={start:0}
    fscore={start:heuristic(start, goal)}
    oheap=[]
    heapq.heappush(oheap, (fscore[start], start))
    while oheap:
        current=heapq.heappop(oheap)[1]
        if current==goal:
            data=[]
            while current in came_from:
                data.append(current)
                current=came_from[current]
            return redressRoad(data)
        close_set.add(current)

        for i, j in neighbors:
            neighbor=current[0]+i, current[1]+j
            tentative_g_score=gscore[current]+heuristic(current, neighbor)
            if 0 <= neighbor[0] < array.shape[0]:
                if 0 <= neighbor[1] < array.shape[1]:
                    if array[neighbor[0]][neighbor[1]]==1:
                        #neighbor is an obstacle
                        continue
                    if i and j:
                        if array[neighbor[0]-i][neighbor[1]]==1 and array[neighbor[0]][neighbor[1]-j]:
                            #diagonal is bounded with obstacles
                            continue
                else:
                    # array bound y walls
                    continue
            else:
                # array bound x walls
                continue

            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                continue

            if  tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1]for i in oheap]:
                came_from[neighbor]=current
                gscore[neighbor]=tentative_g_score
                fscore[neighbor]=tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(oheap, (fscore[neighbor], neighbor))
    return False
