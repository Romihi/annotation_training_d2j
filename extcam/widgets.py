"""extcam ウィンドウの部品: 画像ビュー（拡大・移動・重ね描き・クリック）と地図ビュー（matplotlib）。"""
from __future__ import annotations

import numpy as np
from PyQt5.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QImage, QPainter, QPen
from PyQt5.QtWidgets import QSizePolicy, QWidget

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


def _set_jp_font():
    """地図の日本語（凡例・注記）が豆腐にならないよう、実在するフォントを選ぶ（Windows / Linux 両方）。"""
    import matplotlib
    from matplotlib import font_manager
    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Yu Gothic", "Meiryo", "MS Gothic", "Noto Sans CJK JP", "Noto Serif CJK JP",
                 "IPAGothic", "IPAexGothic", "TakaoGothic", "Hiragino Sans"):
        if name in have:
            matplotlib.rcParams["font.family"] = [name, "sans-serif"]
            break
    matplotlib.rcParams["axes.unicode_minus"] = False


_set_jp_font()


def bgr_to_qimage(img: np.ndarray) -> QImage:
    rgb = np.ascontiguousarray(img[:, :, ::-1])
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()


class ImageView(QWidget):
    """画像を縦横比を保って表示する。ホイールで拡大、右ドラッグで移動、ダブルクリックで全体表示。

    シグナルの座標はすべて画像の画素座標。overlay(painter, to_widget) を差し込んで重ね描きする。
    """
    pressed = pyqtSignal(float, float, int, int)      # x, y, button, modifiers
    dragged = pyqtSignal(float, float)                 # 左ボタンを押したまま動かした位置
    released = pyqtSignal(float, float)
    hovered = pyqtSignal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(320, 180)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self._qimg = None
        self._img_size = (1, 1)
        self._zoom = 1.0
        self._center = None            # 表示中心（画像座標）
        self._pan_from = None
        self._left_down = False
        self.overlays = []             # callable(painter, to_widget)

    # --- 画像 ------------------------------------------------------------
    def set_image(self, img: np.ndarray | None, keep_view: bool = True):
        if img is None:
            self._qimg = None
            self.update()
            return
        size = (img.shape[1], img.shape[0])
        if not keep_view or size != self._img_size or self._center is None:
            self._zoom, self._center = 1.0, (size[0] / 2, size[1] / 2)
        self._img_size = size
        self._qimg = bgr_to_qimage(img)
        self.update()

    def _base_scale(self):
        w, h = self._img_size
        return min(self.width() / max(w, 1), self.height() / max(h, 1))

    def _scale(self):
        return self._base_scale() * self._zoom

    def to_widget(self, x, y) -> QPointF:
        s = self._scale()
        cx, cy = self._center or (0, 0)
        return QPointF(self.width() / 2 + (x - cx) * s, self.height() / 2 + (y - cy) * s)

    def to_image(self, px, py):
        s = self._scale()
        cx, cy = self._center or (0, 0)
        return (cx + (px - self.width() / 2) / s, cy + (py - self.height() / 2) / s)

    def reset_view(self):
        self._zoom = 1.0
        self._center = (self._img_size[0] / 2, self._img_size[1] / 2)
        self.update()

    # --- 描画 ------------------------------------------------------------
    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(30, 30, 30))
        if self._qimg is None:
            p.setPen(QColor(200, 200, 200))
            p.drawText(self.rect(), Qt.AlignCenter, "画像なし")
            return
        w, h = self._img_size
        tl, br = self.to_widget(0, 0), self.to_widget(w, h)
        p.setRenderHint(QPainter.SmoothPixmapTransform, self._zoom < 2.5)
        p.drawImage(QRectF(tl, br), self._qimg)
        p.setRenderHint(QPainter.Antialiasing, True)
        for ov in self.overlays:
            try:
                ov(p, self.to_widget)
            except Exception as e:   # noqa: BLE001  重ね描きの失敗で画面全体を止めない
                print("overlay error:", e)
        p.end()

    # --- 操作 ------------------------------------------------------------
    def wheelEvent(self, e):
        x, y = self.to_image(e.pos().x(), e.pos().y())
        f = 1.25 if e.angleDelta().y() > 0 else 0.8
        self._zoom = float(np.clip(self._zoom * f, 1.0, 20.0))
        # カーソルの下の点が動かないように中心をずらす
        s = self._scale()
        self._center = (x - (e.pos().x() - self.width() / 2) / s, y - (e.pos().y() - self.height() / 2) / s)
        self.update()

    def mouseDoubleClickEvent(self, e):
        if e.button() == Qt.RightButton:
            self.reset_view()

    def mousePressEvent(self, e):
        if e.button() == Qt.RightButton:
            self._pan_from = (e.pos().x(), e.pos().y(), self._center)
            return
        x, y = self.to_image(e.pos().x(), e.pos().y())
        self._left_down = e.button() == Qt.LeftButton
        self.pressed.emit(x, y, int(e.button()), int(e.modifiers()))

    def mouseMoveEvent(self, e):
        if self._pan_from is not None:
            px, py, c = self._pan_from
            s = self._scale()
            self._center = (c[0] - (e.pos().x() - px) / s, c[1] - (e.pos().y() - py) / s)
            self.update()
            return
        x, y = self.to_image(e.pos().x(), e.pos().y())
        if self._left_down:
            self.dragged.emit(x, y)
        else:
            self.hovered.emit(x, y)

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.RightButton:
            self._pan_from = None
            return
        if e.button() == Qt.LeftButton and self._left_down:
            self._left_down = False
            x, y = self.to_image(e.pos().x(), e.pos().y())
            self.released.emit(x, y)


class MapView(FigureCanvas):
    """コースの地図と外形。クリックで地図座標を返す。draw_extra(ax) で重ね描き。"""
    clicked = pyqtSignal(float, float, int)            # x, y [m], button(1=左 3=右)

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(5, 5), tight_layout=True)
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax = self.fig.add_subplot(111)
        self.course = None
        self.draw_extra = None
        self._limits = None
        self.mpl_connect("button_press_event", self._on_click)
        self.mpl_connect("scroll_event", self._on_scroll)

    def set_course(self, course):
        self.course = course
        self._limits = None
        self.redraw()

    def redraw(self, keep_limits: bool = True):
        ax = self.ax
        if keep_limits and self._limits is None and ax.has_data():
            pass
        lim = (ax.get_xlim(), ax.get_ylim()) if keep_limits and self._limits else None
        ax.clear()
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)
        c = self.course
        if c is not None:
            if c.image is not None:
                ax.imshow(c.image, cmap="gray", extent=c.extent(), origin="upper", alpha=0.6, zorder=0)
            for pl in c.polylines():
                ax.plot(pl[:, 0], pl[:, 1], "-", color="#2a7fff", lw=1.2, zorder=2)
            v = c.vertices()
            if len(v):
                ax.plot(v[:, 0], v[:, 1], "o", ms=3.5, color="#2a7fff", zorder=3)
        if self.draw_extra:
            try:
                self.draw_extra(ax)
            except Exception as e:   # noqa: BLE001
                print("map overlay error:", e)
        if lim:
            ax.set_xlim(*lim[0])
            ax.set_ylim(*lim[1])
        elif c is not None and len(c.vertices()):
            v = c.vertices()
            pad = 0.8
            ax.set_xlim(v[:, 0].min() - pad, v[:, 0].max() + pad)
            ax.set_ylim(v[:, 1].min() - pad, v[:, 1].max() + pad)
        self._limits = True
        self.draw_idle()

    def _on_click(self, ev):
        if ev.inaxes != self.ax or ev.xdata is None:
            return
        self.clicked.emit(float(ev.xdata), float(ev.ydata), int(ev.button))

    def _on_scroll(self, ev):
        if ev.inaxes != self.ax or ev.xdata is None:
            return
        f = 0.8 if ev.button == "up" else 1.25
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        cx, cy = ev.xdata, ev.ydata
        self.ax.set_xlim(cx - (cx - x0) * f, cx + (x1 - cx) * f)
        self.ax.set_ylim(cy - (cy - y0) * f, cy + (y1 - cy) * f)
        self.draw_idle()


def pen(color, width=2.0, style=Qt.SolidLine) -> QPen:
    q = QPen(QColor(color))
    q.setWidthF(width)
    q.setStyle(style)
    return q
