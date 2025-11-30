# An implementation of a simple k-NN classifier

# Data type
class Input:
    def __init__(self, p1, p2, p3, res=None):
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.response = res
    
    def distance_to(self, other):
        return ((self.p1 - other.p1) ** 2 + (self.p2 - other.p2) ** 2 + (self.p3 - other.p3) ** 2) ** 0.5

# Dataset
dataset = [
    # --- Cluster 1 (around ~[10,10,10]) → res1 ---
    Input(10, 11, 9, "res1"),
    Input(8, 10, 12, "res1"),
    Input(12, 9, 11, "res1"),
    Input(11, 13, 10, "res1"),
    Input(9, 8, 7, "res1"),

    # --- Cluster 2 (around ~[50,50,50]) → res2 ---
    Input(51, 49, 52, "res2"),
    Input(48, 53, 51, "res2"),
    Input(52, 50, 47, "res2"),
    Input(50, 54, 53, "res2"),
    Input(49, 47, 50, "res2"),

    # --- Cluster 3 (around ~[90,20,70]) → res3 ---
    Input(91, 20, 72, "res3"),
    Input(89, 23, 68, "res3"),
    Input(92, 18, 70, "res3"),
    Input(88, 21, 73, "res3"),
    Input(90, 19, 67, "res3")
]

# k-NN Classifier
def classify(inp):
    correct_res = ""
    nearest = []
    for i in dataset:
        dist = inp.distance_to(i)
        if len(nearest) < 5:
            nearest.append((dist, i.response))
        else:
            max_dist = max(nearest, key=lambda x: x[0])
            if dist < max_dist[0]:
                nearest.remove(max_dist)
                nearest.append((dist, i.response))
    
    votes = {}
    for dist, res in nearest:
        votes[res] = votes.get(res, 0) + 1
    
    correct_res = max(votes, key=votes.get)
    return correct_res

# Testing
print(classify(Input(10, 10, 10)))  # Expected output: "res1"
print(classify(Input(50, 50, 50)))  # Expected output: "res2"
print(classify(Input(70, 33, 59)))  # Expected output: "res3"

# Testing with dataset
correct = 0

for data in dataset:
    predicted = classify(Input(data.p1, data.p2, data.p3))
    print(f"Actual: {data.response}, Predicted: {predicted}")
    if predicted == data.response:
        correct += 1

print(f"Accuracy: {correct}/{len(dataset)} = {correct/len(dataset)*100:.2f}%")