import os, sys

from PyQt5.QtCore import QDir, Qt, QFileInfo, QModelIndex, QDirIterator
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox, QGraphicsScene, QFileSystemModel, \
    QGraphicsPixmapItem
from ui_mainwindow import Ui_MainWindow
from PyQt5.QtGui import QPixmap, QIcon, QImage, QPainter

import torch
import torchvision.transforms as transforms

from timm import create_model
from PIL import Image, ImageQt
import numpy as np

from models import DMemSeg
from list_model import ListModel


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # 项目配置部分框逻辑连接
        self.ui.slider_img.valueChanged.connect(self.slider_img_changed)
        self.ui.spinBox_img.valueChanged.connect(self.ui.slider_img.setValue)

        self.ui.slider_mask.valueChanged.connect(self.slider_mask_changed)
        self.ui.spinBox_mask.valueChanged.connect(self.ui.slider_mask.setValue)

        self.ui.checkBox_hide_img.stateChanged.connect(self.img_state_change)
        self.ui.checkBox_hide_mask.stateChanged.connect(self.mask_state_change)

        self.ui.checkBox_reverse_mask.stateChanged.connect(self.reverse_mask)

        self.ui.action_open_folder.triggered.connect(self.open_folder)
        self.ui.action_save_mask.triggered.connect(self.save_mask)
        self.ui.action_mask_save_as.triggered.connect(self.mask_save_as)
        self.ui.action_save_img.triggered.connect(self.save_img)
        self.ui.action_img_save_as.triggered.connect(self.img_save_as)
        self.ui.action_close.triggered.connect(self.close_folder)
        self.ui.action_quit.triggered.connect(self.close)

        self.ui.action_help.triggered.connect(self.help)

        # 编辑菜单
        self.ui.action_next.triggered.connect(self.next_img)
        self.ui.action_last.triggered.connect(self.last_img)
        self.ui.action_run.triggered.connect(self.run)

        # 视图菜单中所有停靠窗口的显示控制
        self.ui.action_tree.toggled.connect(self.ui.dockWidget_tree.setVisible)
        self.ui.dockWidget_tree.visibilityChanged.connect(self.ui.action_tree.setChecked)
        self.ui.action_config.toggled.connect(self.ui.dockWidget_config.setVisible)
        self.ui.dockWidget_config.visibilityChanged.connect(self.ui.action_config.setChecked)
        self.ui.action_list.toggled.connect(self.ui.dockWidget_list.setVisible)
        self.ui.dockWidget_list.visibilityChanged.connect(self.ui.action_list.setChecked)
        # 调整视图
        self.ui.action_in.triggered.connect(self.zoom_in)
        self.ui.action_out.triggered.connect(self.zoom_out)
        self.ui.action_origin.triggered.connect(self.reset_origin)
        self.ui.action_fit_window.triggered.connect(self.fit_window)
        self.ui.action_fit_width.triggered.connect(self.fit_width)

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            )
        ])

        feature_extractor = create_model(
            'resnet18',
            pretrained=False,
            features_only=True
        )
        self.model = DMemSeg(
            feature_extractor=feature_extractor,
            memory_size=1024,
            feature_channels=[64, 128, 256],
            threshold=0.0005,
            epsilon=1e-12
        )
        self.model.load_state_dict(torch.load('saved_model/map/best_model (1).pt', map_location=torch.device('cpu')))
        self.model.eval()

        self.current_path = ''
        self.scale = 1.0
        self.img = None
        self.mask = None

        self.root_path = ''
        self.paths = []  # 图片文件路径

        # 初始化三个图层
        self.mask_layer = QGraphicsPixmapItem()
        self.mask_layer.setZValue(3)
        self.img_layer = QGraphicsPixmapItem()
        self.img_layer.setZValue(2)
        self.bg_layer = QGraphicsPixmapItem()
        self.bg_layer.setZValue(1)
        # 将图层添加到场景
        self.scene = QGraphicsScene()
        self.scene.addItem(self.mask_layer)
        self.scene.addItem(self.img_layer)
        self.scene.addItem(self.bg_layer)
        self.ui.graphicsView.setScene(self.scene)

        self.file_model = None
        self.ui.treeView.setHeaderHidden(True)
        self.ui.treeView.clicked.connect(self.select)

        self.list_model = None
        self.ui.listView.clicked.connect(self.select)

        # self.ui.treeView.clicked.connect(self.select)

    def bg_img(self):
        return Image.fromarray(np.zeros((*reversed(self.img.size), 3)), mode='RGB')

    def select(self, select_index: QModelIndex):
        if select_index:
            sender = self.sender()
            if sender is self.ui.treeView:
                path = self.file_model.filePath(select_index)
                if QFileInfo(path).isFile():
                    # ListView 设置选中
                    self.list_select(path)
                    self.set_file(path)
            elif sender is self.ui.listView:
                path = self.list_model.item_path(select_index)
                self.tree_select(path)
                self.set_file(path)
            else:
                print(sender)

    def tree_select(self, path):
        tree_index = self.file_model.index(path)
        self.ui.treeView.setCurrentIndex(tree_index)
        while tree_index.isValid():
            self.ui.treeView.setExpanded(tree_index, True)
            tree_index = tree_index.parent()

    def list_select(self, path):
        list_index = self.list_model.index_of(path)
        self.ui.listView.setCurrentIndex(list_index)

    def set_file(self, path):
        self.current_path = path
        self.img = Image.open(self.current_path).convert('RGB')
        self.mask = None
        self.img_layer.setPixmap(ImageQt.toqpixmap(self.img))
        self.mask_layer.setPixmap(QPixmap())
        self.bg_layer.setPixmap(ImageQt.toqpixmap(self.bg_img()))
        self.update_opacity()

    def open_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, '打开文件夹', QDir.currentPath())
        if folder_path:
            self.root_path = folder_path
            self.file_model = QFileSystemModel()
            self.file_model.setNameFilters(['*.png', '*.jpg', '*.gif', '*.bmp'])
            self.file_model.setNameFilterDisables(False)  # 启用过滤器
            # root = 'E:/project/科研竞赛/晶圆缺陷检测/数据集/wafer_segmentation/product/test'
            self.setWindowTitle(self.root_path)
            self.file_model.setRootPath(self.root_path)
            self.file_model.directoryLoaded.connect(self.dir_loaded)

            self.ui.treeView.setModel(self.file_model)
            self.ui.treeView.setRootIndex(self.file_model.index(self.root_path))
            for i in range(1, self.file_model.columnCount()):
                self.ui.treeView.hideColumn(i)

            self.img = None
            self.mask = None
            self.paths = []
            self.mask_layer.setPixmap(QPixmap())
            self.img_layer.setPixmap(QPixmap())
            self.bg_layer.setPixmap(QPixmap())
            self.ui.listView.setModel(None)

    def default_mask_path(self):
        basename = os.path.basename(self.current_path)
        filename, extension = os.path.splitext(basename)
        return self.current_path.replace(extension, '_mask' + extension)

    def save_mask(self):
        if self.current_path and self.mask:
            self.mask.convert('RGB').save(self.default_mask_path())

    def mask_save_as(self):
        if self.current_path and self.mask:
            mask_path = self.default_mask_path()
            save_path, _ = QFileDialog.getSaveFileName(self, '掩码文件另存为', mask_path,
                                                       'All Files (*);;Text Files (*.txt)')
            if save_path:
                self.mask.convert('RGB').save(save_path)

    def default_img_path(self):
        basename = os.path.basename(self.current_path)
        filename, extension = os.path.splitext(basename)
        return self.current_path.replace(extension, '_mixed' + extension)

    def get_mixed_image(self):
        image = QImage(self.scene.sceneRect().size().toSize(), QImage.Format_ARGB32)
        image.fill(Qt.transparent)
        painter = QPainter(image)
        self.scene.render(painter)
        painter.end()
        return image

    def save_img(self):
        if self.current_path and self.mask:
            image = self.get_mixed_image()
            ImageQt.fromqimage(image).save(self.default_img_path())

    def img_save_as(self):
        if self.current_path and self.mask:
            img_path = self.default_img_path()
            save_path, _ = QFileDialog.getSaveFileName(self, '图像文件另存为', img_path,
                                                       'All Files (*);;Text Files (*.txt)')
            if save_path:
                image = self.get_mixed_image()
                ImageQt.fromqimage(image).save(save_path)

    def close_folder(self):
        if self.root_path:
            self.root_path = ''
            self.img = None
            self.mask = None
            self.paths = []
            self.setWindowTitle('')
            self.ui.treeView.setModel(None)
            self.ui.listView.setModel(None)

            self.mask_layer.setPixmap(QPixmap())
            self.img_layer.setPixmap(QPixmap())
            self.bg_layer.setPixmap(QPixmap())

    def help(self):
        content = """
软件使用说明：
1. 打开软件：运行程序即可打开界面。
2. 选择文件夹：点击菜单栏中的 "文件" -> "打开文件夹"，选择图片文件所在的文件夹。
3. 选择图片：在左侧文件列表中选择图片文件，中央区域将显示该图片。
4. 调整视图：使用工具栏按钮调整图像显示效果。
5. 执行检测：点击执行按钮进行缺陷检测。
6. 观察结果：检测完成后，可观察缺陷掩码，调整透明度等参数。
7. 保存结果：点击菜单栏中的 "文件" -> "保存掩码"、“保存图像”，选择保存路径保存结果。
8. 关闭软件：点击菜单栏中的 "文件" -> "退出"，关闭软件。"""
        QMessageBox.information(self, '帮助', content)

    def dir_loaded(self, dir_path):
        dir_index = self.file_model.index(dir_path)
        for row in range(self.file_model.rowCount(dir_index)):
            index = self.file_model.index(row, 0, dir_index)
            file_path = self.file_model.filePath(index)
            # 如果是文件，则将其路径添加到列表中
            if QFileInfo(file_path).isFile():
                if file_path not in self.paths:
                    self.paths.append(file_path)
            # 如果是目录，则异步获取该目录
            elif QFileInfo(file_path).isDir():
                self.file_model.fetchMore(index)

        self.list_model = ListModel(self.paths)
        self.ui.listView.setModel(self.list_model)

    def next_img(self):
        if self.paths is None:
            return
        if not self.current_path:
            path = self.paths[0]
        else:
            p = self.paths.index(self.current_path) + 1
            if p < len(self.paths):
                path = self.paths[p]
            else:
                return
        self.list_select(path)
        self.tree_select(path)
        self.set_file(path)

    def last_img(self):
        if self.paths is None:
            return
        if not self.current_path:
            path = self.paths[0]
        else:
            p = self.paths.index(self.current_path) - 1
            if p > 0:
                path = self.paths[p]
            else:
                return
        self.list_select(path)
        self.tree_select(path)
        self.set_file(path)

    def run(self):
        if self.current_path:
            list_index = self.list_model.index_of(self.current_path)
            pixmap = QPixmap(self.current_path)
            if not pixmap.isNull():
                with torch.no_grad():
                    predict, _ = self.model(self.transform(self.img).unsqueeze(0))
                    predict = predict.squeeze()
                    mask_tensor = torch.where(predict[0] > predict[1], torch.tensor(0), torch.tensor(255))
                    mask_array = mask_tensor.numpy().astype(dtype=np.uint8)
                    self.mask = Image.fromarray(mask_array, mode='L')
                    # 与原始图像大小统一
                    self.mask = self.mask.resize(self.img.size, resample=Image.LANCZOS)
                    # 恢复为黑白图
                    self.mask = self.mask.point(lambda x: 0 if x < 128 else 255)
                    gray_array = np.asarray(self.mask)
                    # 灰度图 -> RGBA
                    mask_array = np.repeat(gray_array[:, :, np.newaxis], 4, axis=2)
                    if self.ui.checkBox_reverse_mask.isChecked():
                        mask_array[:, :, 3] = 255 - mask_array[:, :, 0]
                    else:
                        mask_array[:, :, 3] = mask_array[:, :, 0]
                    self.mask = Image.fromarray(mask_array, mode='RGBA')
                    self.img_layer.setPixmap(ImageQt.toqpixmap(self.img))
                    self.mask_layer.setPixmap(ImageQt.toqpixmap(self.mask))
                    self.update_opacity()
                    # 修改 ListView 中的 CheckState
                    self.list_model.finished(list_index)

    def zoom_in(self):
        self.scale += 0.1
        self.ui.graphicsView.scale(self.scale, self.scale)

    def zoom_out(self):
        self.scale -= 0.1
        self.ui.graphicsView.scale(self.scale, self.scale)

    def reset_origin(self):
        self.scale = 1.0
        self.ui.graphicsView.resetTransform()

    def fit_window(self):
        self.ui.graphicsView.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

    def fit_width(self):
        self.ui.graphicsView.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatioByExpanding)

    def reverse_mask(self, state):
        if self.mask is not None:
            mask_array = np.asarray(self.mask).copy()
            if state == Qt.Checked:
                mask_array[:, :, 3] = 255 - mask_array[:, :, 0]
            else:
                mask_array[:, :, 3] = mask_array[:, :, 0]
            self.mask = Image.fromarray(mask_array, mode='RGBA')
            self.mask_layer.setPixmap(ImageQt.toqpixmap(self.mask))
            self.update_opacity()

        # 视图区域重绘方法

    def update_opacity(self):
        alpha_img = self.ui.spinBox_img.value() / 100.0
        alpha_mask = self.ui.spinBox_mask.value() / 100.0
        self.mask_layer.setOpacity(alpha_mask)
        self.img_layer.setOpacity(alpha_img)
        self.scene.setSceneRect(self.img_layer.boundingRect())
        self.scene.update()

    def slider_img_changed(self, val):
        self.ui.spinBox_img.setValue(val)
        if val == 0:
            self.ui.checkBox_hide_img.setCheckState(Qt.Checked)
        else:
            self.ui.checkBox_hide_img.setCheckState(Qt.Unchecked)
        self.update_opacity()

    def slider_mask_changed(self, val):
        self.ui.spinBox_mask.setValue(val)
        if val == 0:
            self.ui.checkBox_hide_mask.setCheckState(Qt.Checked)
        else:
            self.ui.checkBox_hide_mask.setCheckState(Qt.Unchecked)
        self.update_opacity()

    def img_state_change(self, state):
        if state == Qt.Checked:
            self.ui.spinBox_img.setValue(0)
        elif state == Qt.Unchecked and self.ui.spinBox_img.value() == 0:
            self.ui.spinBox_img.setValue(100)

    def mask_state_change(self, state):
        if state == Qt.Checked:
            self.ui.spinBox_mask.setValue(0)
        elif state == Qt.Unchecked and self.ui.spinBox_mask.value() == 0:
            self.ui.spinBox_mask.setValue(100)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon(':/icon/tools.png'))
    mainwindow = MainWindow()
    mainwindow.show()
    sys.exit(app.exec())
