from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np

class CentroidTracker():
    def __init__(self, maxDisappeared=50, maxDistance=50):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        # Track number of frames since object last seen
        self.disappeared = OrderedDict()
        # Maximum consecutive frames a given object is allowed before
        # being marked as disappeared
        self.maxDisappeared = maxDisappeared
        # Maximum distance between centroid that can still be registered to an object
        self.maxDistance = maxDistance

    def register(self, centroid):
        # Register an object using the next available ID
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def unregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        # Nothing seen, update disappeared objects and return
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.unregister(objectID)
            return self.objects
        inputCentroids = np.zeros((len(rects), 2), dtype=int)
        # For each bounding box, find their centroid
        for i, (x1, y1, x2, y2) in enumerate(rects):
            cX = int((x1 + x2) / 2.0)
            cY = int((y1 + y2) / 2.0)
            inputCentroids[i] = (cX, cY)
        # Not tracking anything, register centroids and return
        if len(self.objects) == 0:
            for i in range(len(inputCentroids)):
                self.register(inputCentroids[i])
            return self.objects
        # There are objects being tracked, try to match centroids to registered objects
        objectIDs = list(self.objects.keys())
        objectCentroids = list(self.objects.values())

        D = dist.cdist(np.array(objectCentroids), inputCentroids)

        # in order to perform this matching we must (1) find the
        # smallest value in each row and then (2) sort the row
        # indexes based on their minimum values so that the row
        # with the smallest value is at the *front* of the index
        # list
        rows = D.min(axis=1).argsort()
        # next, we perform a similar process on the columns by
        # finding the smallest value in each column and then
        # sorting using the previously computed row index list
        cols = D.argmin(axis=1)[rows]

        # in order to determine if we need to update, register,
        # or deregister an object we need to keep track of which
        # of the rows and column indexes we have already examined
        usedRows = set()
        usedCols = set()

        for (row, col) in zip(rows, cols):
            # If we already examined the row or column before, ignore
            if row in usedRows or col in usedCols:
                continue
            # If distance is greater than allowed, do not associate them
            if D[row, col] > self.maxDistance:
                continue
            # Otherwise, grab the object ID for the current row,
            # set its new centroid, reset the disappeared counter
            objectID = objectIDs[row]
            self.objects[objectID] = inputCentroids[col]
            self.disappeared[objectID] = 0
            # indicate that we have examined each of the row and column indexes
            usedRows.add(row)
            usedCols.add(col)

        unusedRows = set(range(0, D.shape[0])).difference(usedRows)
        unusedCols = set(range(0, D.shape[1])).difference(usedCols)

        # in the event that the number of object centroids is
        # equal or greater than the number of input centroids
        # we need to check and see if some of these objects have
        # potentially disappeared
        if D.shape[0] >= D.shape[1]:
            # loop over the unused row indexes
            for row in unusedRows:
                # grab the object ID for the corresponding row
                # index and increment the disappeared counter
                objectID = objectIDs[row]
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.unregister(objectID)
        # else new objects have appeared
        else:
            for col in unusedCols:
                self.register(inputCentroids[col])

        return self.objects

