from typing import List

from PyQt5.QtCore import QAbstractListModel, Qt, QModelIndex


class ListModel(QAbstractListModel):
    def __init__(self, paths: List[str], parent=None):
        super(ListModel, self).__init__(parent)
        self.items = [{
            'path': path,
            'done': False
        } for path in paths]

    def data(self, index: QModelIndex, role: int = ...):
        row = index.row()
        if index.isValid() or 0 <= row < len(self.items):
            item = self.items[row]
            if role == Qt.DisplayRole:
                return item['path']
            elif role == Qt.CheckStateRole:
                return Qt.Checked if item['done'] else Qt.Unchecked
        return None

    def rowCount(self, parent=QModelIndex()):
        return len(self.items)

    def item_path(self, index: QModelIndex):
        return self.items[index.row()]['path']

    def index_of(self, path):
        return self.index([item['path'] for item in self.items].index(path))

    def finished(self, index: QModelIndex):
        self.items[index.row()]['done'] = True
