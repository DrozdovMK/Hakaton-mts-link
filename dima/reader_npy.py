import numpy as np

print(f'метки кластеров: {type(np.load("метки_кластеров.npy"))}\n')

clusters = list(np.load("метки_кластеров.npy"))



# print(f'ОТветы сотрудников: {np.load("ответы_сотрудников.npy")}\n')
# print(f'Эмбединги: {np.load("эмбеддинги2d.npy")}\n')

print(f'Эмбединги: {np.load("эмбеддинги2d.npy")}\n')
embeddings = list(np.load("эмбеддинги2d.npy"))

# посичтали количесвто каждого числа в кластере
def count_numbers(numbers):
  counts = {}
  for number in numbers:
    if number in counts:
      counts[number.item()]["size"] += 1
    #   counts[number]["xy"].append(embeddings[i])
    else:
      counts[number.item()] = {}
      counts[number.item()]["size"] = 1
    #   counts[number]["xy"] = [embeddings[i]]
  return counts


print(count_numbers(clusters))
data = count_numbers(clusters)

# заполнили для каждого кластера всевозможные координаты
def add_coordinats(data, clusters, embeddings):
    for i, embedding in enumerate(embeddings):
        if "xy" in data[clusters[i].item()]:
            data[clusters[i].item()]["xy"].append(embedding)
        else:
            data[clusters[i].item()]["xy"] = [embedding]

    return data

print(add_coordinats(data, clusters, embeddings))
data = add_coordinats(data, clusters, embeddings)

# для каждой координаты найдем центр
def center_of_coordinates(data):
    for clust in data:
    #    print(data[clust])
    #    print(data[clust]["xy"])
    #    print(np.mean(data[clust]["xy"], axis=0).tolist()[0])
       data[clust]["x"] = np.mean(np.array(data[clust]["xy"]), axis=0).tolist()[0]
       data[clust]["y"] = np.mean(np.array(data[clust]["xy"]), axis=0).tolist()[1]
       data[clust].pop("xy")

    return data

print('\n\n\n')
# print(center_of_coordinates(data))
data = center_of_coordinates(data)

answers = np.load("ответы_сотрудников.npy")
def add_text_value(data, clustesr, answers):
    for i, clust in enumerate(clusters):
        if "text" in data[clustesr[i].item()]:
            data[clustesr[i].item()]["text"].append(answers[i].item())
        else:
            data[clustesr[i].item()]["text"] = [answers[i].item()]

    return data
    
data = add_text_value(data, clusters, answers)
print('\n\n\n')
print(data)
  

data_plotly = {
    "деньги": {"x": 10, "y": 12, "size": 36, "text": ["мани", "бабки"]},
    "интерес": {"x": 14, "y": 5, "size": 50, "text": ["интерес1", "интерес2"]},
    "время": {"x": 30, "y": 20, "size": 100, "text": ["секунда", "часы"]}
}