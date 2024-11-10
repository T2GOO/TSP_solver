#Solve a traveling salesman problem using K-means clustering recursively
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import itertools, math, random, time, csv

DEF_NB_CLUSTERS = 6
DEF_THRESHOLD = 8

# Class to represent a city
class City:

    def __init__(self, coordinates, name= None):
        # Coordinates of the city
        self.coordinates = coordinates
        if name==None:
            self.name = f"City_{self.coordinates[0]}{self.coordinates[1]}"
        else:
            self.name = name

    def showTime(self):
        print(f"City name: {self.name}, coordinates: ({self.coordinates[0]}, {self.coordinates[1]})")

# Class to represent a cluster (geographic area)
class Area(City):

    def __init__(self, coordinates, cities = None,  nbClusters = DEF_NB_CLUSTERS, isEarth = False, threshold = DEF_THRESHOLD):
        # Coordinates of the area (mean of all cities in the area)
        super().__init__(coordinates)
        # List of cities in the area
        if cities == None:
            self.citiesList = []
        else:
            self.citiesList = cities
        # List of (sub)areas in the area
        self.areasList = []
        # number of sub-areas in the area
        self.nbMAreas = nbClusters
        self.threshold = threshold
        # Is earth .
        self.isEarth = isEarth
        if self.isEarth:
            # next cluster
            self.nextCluster = self.citiesList[0]
            self.nextClusterCoordinates = self.citiesList[0].coordinates
            # previous cluster
            self.previousCluster = self.citiesList[0]
            self.previousClusterCoordinates = self.citiesList[0].coordinates
        else:
            # Set during the execution
            # next cluster
            self.nextCluster = None
            self.nextClusterCoordinates = [0, 0]
            # previous cluster
            self.previousCluster = None
            self.previousClusterCoordinates = [0, 0]
        
        # Frist sub-area
        self.firstSubCluster = None
        # Last sub-area
        self.lastSubCluster = None
        
    def addCities(self, cities):
        self.citiesList.extend(cities)

    def setNextClusters(self, nextC):
        self.nextCluster = nextC
        self.nextClusterCoordinates = nextC.coordinates

    def setPreviousClusters(self, previousC):
        self.previousCluster = previousC
        self.previousClusterCoordinates = previousC.coordinates
    
    # From the cities in the area, get clusters (sub-areas)
    # Add cities to their sub-areas
    # Get the first and last clusters
    # Sort clusters
    # Output: self.areasList = list of sub-areas (sorted)
    def getClusters(self):
        data = []
        # Check if there are enough cities to cluster
        #if len(self.citiesList) >= TRESHOLD:
        # Add coordinates of all cities in the area to the data
        for city in self.citiesList:
            data.append([city.coordinates[0], city.coordinates[1]])

        # Cluster the data
        self.kmeans = KMeans(n_clusters=self.nbMAreas)
        self.kmeans.fit(data)

        # Add each cluster to areasList
        for center in self.kmeans.cluster_centers_:
            self.areasList.append(Area( coordinates = center, nbClusters=self.nbMAreas, threshold = self.threshold))

        # Add cities to each cluster
        for i, label in enumerate(self.kmeans.labels_):
            self.areasList[label].addCities([self.citiesList[i]])

        # Remove empty clusters
        self.areasList = list(filter(lambda x: len(x.citiesList) > 0, self.areasList))

        # Sort clusters to get the minimum distance between clusters
        self.getPerfectPath()

        # get first and last sub-cluster
        self.getFirstAndLastSubCluster()

        # set sub-clusters neighbours
        for i in range(len(self.areasList)):
            nextClustIndex = i+1
            if nextClustIndex == len(self.areasList):
                self.areasList[i].setNextClusters(self.nextCluster)
            else:
                self.areasList[i].setNextClusters(self.areasList[nextClustIndex])
            previousClustIndex = i-1
            if previousClustIndex == -1:
                self.areasList[i].setPreviousClusters(self.previousCluster)
            else:
                self.areasList[i].setPreviousClusters(self.areasList[previousClustIndex])

    # Get the perfect path between points (cities or clusters)
    # First sub-cluster is firstSubCluster
    # Last sub-cluster is lastSubCluster
    # Output: self.areasList = list of sub-areas (sorted)
    # Use permutations
    def getPerfectPath(self, cluster = True):
        # Get the list of sub-areas without the first and last sub-areas
        if cluster: localList = self.areasList
        else: localList = self.citiesList

        minDistance = 999999999999999
        # Get previous point = lastClusterCoordinate from previous cluster
        try:
            previousPoint = self.previousCluster.lastSubCluster
        except:
            previousPoint = self.previousCluster
        # Get next point = next cluster coordinate
        nextPoint = self.nextCluster
        # Get for all pemrutations
        for permutation in itertools.permutations(localList):
            permutation = list(permutation)
            # Calculate the total ditance
            permutation.insert(0, previousPoint)
            permutation.append(nextPoint)
            distance = self.distanceFullPath(permutation)
            if distance < minDistance:
                minDistance = distance
                if cluster:
                    self.areasList = permutation[1:-1]
                else:
                    self.citiesList = permutation[1:-1]

    @staticmethod
    def distanceFullPath(myClusters):
        distance = 0
        for i in range(len(myClusters) - 1):
            distance += math.dist(myClusters[i].coordinates, myClusters[i+1].coordinates)
        return distance

    # Function returning the optimal path (cities) --> function call by user
    def getOptimalPath(self):
        listOfCities = []
        # Check if there are enough cities in cluster
        if len(self.citiesList) >= self.threshold:
            # Get sub-clusters
            self.getClusters()
            # for each sub-cluster, get the optimal path
            for area in self.areasList:
                listOfCities.extend(area.getOptimalPath())
            if self.isEarth:
               # Add the first city at the end (to close the path)
                listOfCities.append(listOfCities[0])
            return listOfCities
        # If not enough cities, return the perfect path between cities in the area
        else:
            self.perfectCitiesPath()
            return self.citiesList

    def perfectCitiesPath(self):
        # Check if there are not too many cities in cluster
        if len(self.citiesList) < self.threshold:
            # Sort clusters to get the minimum distance between clusters
            self.getPerfectPath(False)
            # Get first and last cities
            self.getFirstAndLastSubCluster(False)
        else:
            print("ERROR: Too many cities")

    def getFirstAndLastSubCluster(self, clusters=True):
        if clusters:
            self.firstSubCluster = self.areasList[0]
            self.lastSubCluster = self.areasList[-1]
        else:
            self.firstSubCluster = self.citiesList[0]
            self.lastSubCluster = self.citiesList[-1]

    def showTime(self):
        print(f"Cluster name: {self.name}, coordinates: ({self.coordinates[0]}, {self.coordinates[1]})")
        # Print the list of cities in the cluster
        for city in self.citiesList:
            city.showTime()

    # Display the area
    def displayArea(self, solution = False, showInfo = True):
        x = []
        y = []
        for city in self.citiesList:
            x.append(city.coordinates[0])
            y.append(city.coordinates[1])
        # True if clusters have been solved
        if solution:
            try:
                # Plot cities with clusters colors
                plt.scatter(x, y, c=self.kmeans.labels_)
            except:
                print("Please solve the problem first")
        # False if clusters have not been solved
        else:
            # Plot cities without clusters colors
            plt.scatter(x, y)
        if showInfo:
            # add annotations for each point
            for city in self.citiesList:
                plt.annotate(f'{city.name}({city.coordinates[0]},{city.coordinates[1]})', xy=(city.coordinates[0], city.coordinates[1]), textcoords='offset points', xytext=(0, 10), ha='center')
        plt.show()
        
def displayConnectedCities(citiesList, plot = True, routes = True, color = "black"):
    # Check if the list is not empty
    if not citiesList:
        print("The cities list is empty.")
        return
    if len(citiesList) <= 200:
        marker = 'o'
    else:
        marker = ''
    # Extract coordinates and names for plotting and annotation
    x_coords = [city.coordinates[0] for city in citiesList]
    y_coords = [city.coordinates[1] for city in citiesList]
    names = [city.name for city in citiesList]

    if routes:
    # Connect cities (black dots)
        plt.plot(x_coords, y_coords, marker=marker, color = color)  # Connects in order
    else:
        plt.scatter(x_coords, y_coords, marker=marker, color = color)
    
    # Annotate each city
    # for i, name in enumerate(names):
    #     plt.annotate(name, (x_coords[i], y_coords[i]))
    
    # Display the graph
    if plot:
        plt.show()

def importData(filePath):
    data = []
    with open(filePath) as fp:
        line = fp.readline()
        while line:
            data.append(line)
            line = fp.readline()
    # Delete 7 first first lines
    data = data[6:-2]
    # from "A BBBB.BBB CCCC.CCC" to [BBBB.BBB, CCCC.CCC]
    for line in range(len(data)):
        strline = data[line].split(" ")
        data[line] = [float(strline[2][:-2]), float(strline[1])]
    return data

def generateCities(numOfCities):
    # list of cities 
    cities = []
    # Minimum dimension of the map (minDim x minDim)
    minDim = numOfCities
    # List storing coordinates
    coorList = []
    # Randomly generate cities
    for i in range(numOfCities):
        # Generate random coordinates
        randCoor = [random.uniform(0,minDim), random.uniform(0,minDim)]
        # Check if the coordinates are not already in the list
        while randCoor in coorList:
            # If the coordinates are already in the list, generate a new one
            randCoor = [random.uniform(0,minDim), random.uniform(0,minDim)]
        # Add the coordinates to the list
        coorList.append(randCoor)
        # Add the city to the list
        cities.append(City(randCoor))
    # Return the list of cities
    return cities

def run_single(nb_clust :int = 5, tresh:int = 7, option:int = 1, param = '', display_results : bool = True):
    nbClust = nb_clust
    threshold = tresh

    if nbClust > tresh:
        print("nb_clust must be < than tresh")
        return

    match option:
        case 1 :
            # list of cities (FIXED)
            cities = [
                City([0,5], name = "Paris"),
                City([10,2], name = "London"),
                City([8,10], name = "Berlin"),
                City([7,10], name = "Milan"),
                City([3,4], name = "Rio"),
                City([5,4], name = "Sydney"),
                City([12,5], name = "Moscow"),
                City([1,6], name = "Seoul"),
                City([1,5], name = "Tokyo"),
                City([5,8], name = "NewYork")]
        case 2:
            if type(param) is int:
                # Random cities
                cities = generateCities(param)
            else:
                print("For option 2, param must be an integer")
                return   
        case 3:
            # Import cities from data
            cities = []
            try:
                data = importData(param)
            except Exception as E:
                print("For option 3,", E)
                return
            for i in range(len(data)):
                cities.append(City(data[i]))
        case 20:
            # Case 2 but for multi generation
            cities = param
        case _:
            print("Option not handle. Must be 1, 2 or 3")
            return

    earth = Area([0,0], cities = cities, isEarth = True, nbClusters = nbClust, threshold = threshold)
    start_time = time.time()
    optimal_path = earth.getOptimalPath()
    execution_time = time.time() - start_time
    distance = Area.distanceFullPath(optimal_path)
    print("Full path distance:",distance)
    print("--- %s seconds ---" % execution_time)
    displayConnectedCities(optimal_path, plot=display_results, routes=True)
    return execution_time, distance, optimal_path

def run_multi(iterations : int = 10, nb_clust :int = 5, tresh:int = 7, option:int = 1, param = '', display_results : bool = True):

    if option == 2:
        if type(param) is int:
            # Random cities
            cities = generateCities(param)
        else:
            print("For option 2, param must be an integer")
            return   
        
    res_file_name = 'resutlts_'+ str(int(time.time())) + '.csv'
    with open(res_file_name, 'a', newline='') as sheet:
        writer = csv.writer(sheet)
        writer.writerow(['iteration',
                        'nb clust',
                        'tresh',
                        'execution time',
                        'distance'])
        for i in range(iterations):
            if option == 2:
                execution_time, distance, optimal_path = run_single(nb_clust, tresh, 20, cities, display_results = False)
            else:
                execution_time, distance, optimal_path = run_single(nb_clust, tresh, option, param, display_results = False)
            print("- Iteration", i)
            # Write in excel sheet the results
            writer.writerow([i,
                             nb_clust,
                             tresh,
                             execution_time,
                             distance])
            displayConnectedCities(optimal_path, plot=False)
    if display_results:
        plt.show()


if __name__ == "__main__":
    # Option 1 -> no param, run 10 default cities
    run_single(nb_clust = 5, tresh = 7, option = 1)

    # Option 2 -> param = number of cities to generate
    run_single(nb_clust = 5, tresh = 7, option = 2, param = 50)

    # Option 3 -> param = path the data file
    run_single(nb_clust = 5, tresh = 7, option = 3, param = "data/berlin52.tsp")

    # Run several iteration (same options as run_single + iterations)
    run_multi(iterations = 50, nb_clust = 5, tresh = 7, option = 2, param = 50)

    