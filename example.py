from food.classify.function import decision_tree

if __name__ == "__main__":
    file_path = 'food/data/data.csv'  # 替换为你的文件路径
    print ("使用decision_tree")
    decision_tree(file_path)