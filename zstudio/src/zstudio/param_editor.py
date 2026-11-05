"""QGroundControl-style parameter editor for OpenVINS/Kalibr YAML and VOXL JSON configs.

Left: files and groups with modified counts. Top: live search over names,
descriptions and values, plus a "Modified only" filter. Table: Name, Value,
Default, Description with type-aware inline editors (checkbox, spin boxes that
accept scientific notation, enum combo boxes, validated list/matrix text).
Edits are validated by ``params.Param.coerce``; rejected input leaves the value
unchanged and explains why in the status line. Saving rewrites only the edited
value spans, so comments and formatting are kept.
"""

import math
import re
from pathlib import Path

from PyQt5.QtCore import (
    QAbstractTableModel,
    QEvent,
    QModelIndex,
    QObject,
    Qt,
    QTimer,
    QUrl,
    pyqtSignal,
)
from PyQt5.QtGui import QColor, QDesktopServices, QKeySequence
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QHeaderView,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QStyledItemDelegate,
    QTableView,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .params import ParamSet, value_text, _format_float, _FLOAT, _INT
from .widgets import AMBER, LINE, MUTED, TEAL, TEXT, font, icon, label

RED = "#f3a6a6"
_TABLE_STYLE = f"""
QTableView {{ background:#101923; alternate-background-color:#16212b; border:1px solid {LINE}; border-radius:6px;
             gridline-color:{LINE}; selection-background-color:#25483e; }}
QTableView::item {{ padding:4px 8px; }}
"""


def _button(text, callback, symbol=None, primary=False):
    widget = QPushButton(text)
    if symbol:
        widget.setIcon(icon(symbol, "#10251f" if primary else MUTED))
    if primary:
        widget.setObjectName("primary")
    widget.clicked.connect(callback)
    return widget


class ScientificSpinBox(QDoubleSpinBox):
    """A double spin box that shows the shortest exact text and accepts 1e-12 style input."""

    _partial = re.compile(r"[-+]?(\d+\.?\d*|\.\d*)?([eE][-+]?\d*)?\Z")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDecimals(323)
        self.setRange(-1e300, 1e300)
        self.setKeyboardTracking(False)

    def textFromValue(self, value):
        return _format_float(value)

    def valueFromText(self, text):
        try:
            return float(text)
        except ValueError:
            return self.value()

    def validate(self, text, position):
        from PyQt5.QtGui import QValidator

        stripped = text.strip()
        if _FLOAT.match(stripped) or _INT.match(stripped):
            value = float(stripped)
            if self.minimum() <= value <= self.maximum():
                return QValidator.Acceptable, text, position
            return QValidator.Intermediate, text, position
        if not stripped or self._partial.match(stripped):
            return QValidator.Intermediate, text, position
        return QValidator.Invalid, text, position

    def fixup(self, text):
        return self.textFromValue(self.value())

    def stepBy(self, steps):
        value = self.value()
        magnitude = 10 ** (math.floor(math.log10(abs(value))) - 1) if value else 0.1
        self.setValue(value + steps * magnitude)


class ParamTableModel(QAbstractTableModel):
    COLUMNS = ("Name", "Value", "Default", "Description")
    rejected = pyqtSignal(str)
    edited = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.pset = None
        self.rows = []  # [(display key, Param)]
        self.error_key = None

    def set_rows(self, pset, rows):
        self.beginResetModel()
        self.pset, self.rows = pset, rows
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.rows)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.COLUMNS)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self.COLUMNS[section]
        return None

    def param(self, row):
        return self.rows[row][1]

    def flags(self, index):
        if not index.isValid():
            return Qt.NoItemFlags
        flags = Qt.ItemIsEnabled | Qt.ItemIsSelectable
        p = self.param(index.row())
        if index.column() == 1 and not p.readonly:
            flags |= Qt.ItemIsUserCheckable if p.type == "bool" else Qt.ItemIsEditable
        return flags

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        key, p = self.rows[index.row()]
        column = index.column()
        if role == Qt.DisplayRole:
            if column == 0:
                return key + ("  •" if p.dirty else "")
            if column == 1:
                if p.type == "bool":
                    return "true" if p.value else "false"
                return p.text() + (f"  {p.unit}" if p.unit else "")
            if column == 2:
                return p.text(p.default)
            return (p.description or p.notes).split("\n")[0]
        if role == Qt.EditRole:
            return p.value
        if role == Qt.CheckStateRole and column == 1 and p.type == "bool":
            return Qt.Checked if p.value else Qt.Unchecked
        if role == Qt.ForegroundRole:
            if column == 0 and p.modified:
                return QColor(AMBER)
            if column in (2, 3):
                return QColor(MUTED)
            if column == 1 and p.readonly:
                return QColor(MUTED)
            return QColor(TEXT)
        if role == Qt.FontRole:
            f = font(10, bold=column == 0 and p.modified, mono=column in (1, 2))
            if column == 0 and p.dirty:
                f.setItalic(True)
            return f
        if role == Qt.BackgroundRole:
            if key == self.error_key:
                return QColor(110, 40, 46, 120)
            if p.dirty:
                return QColor(239, 185, 140, 26)
        if role == Qt.ToolTipRole:
            return self._tooltip(key, p)
        return None

    @staticmethod
    def _tooltip(key, p):
        lines = [f"<b>{key}</b> &nbsp;<span style='color:{MUTED}'>{p.type}</span>"]
        if p.description:
            lines.append(p.description)
        if p.notes and p.notes != p.description:
            lines.append(
                f"<span style='color:{MUTED}'>{p.notes.replace(chr(10), '<br>')}</span>"
            )
        facts = []
        if p.choices:
            facts.append("choices: " + ", ".join(p.choices))
        if p.minimum is not None or p.maximum is not None:
            lo = value_text(p.minimum) if p.minimum is not None else "-inf"
            hi = value_text(p.maximum) if p.maximum is not None else "inf"
            facts.append(f"range: [{lo}, {hi}]")
        if p.unit:
            facts.append(f"unit: {p.unit}")
        facts.append(f"default: {p.text(p.default)}")
        if p.file is not None:
            facts.append(f"{p.file.name}:{p.line + 1}")
        lines.append(f"<span style='color:{MUTED}'>{' · '.join(facts)}</span>")
        return "<br>".join(lines)

    def setData(self, index, value, role=Qt.EditRole):
        if not index.isValid() or index.column() != 1 or self.pset is None:
            return False
        key, p = self.rows[index.row()]
        if role == Qt.CheckStateRole:
            value = value == Qt.Checked or value is True or value == 2
        elif role != Qt.EditRole:
            return False
        try:
            self.pset.set(p, value)
        except ValueError as error:
            self.error_key = key
            self.rejected.emit(str(error))
            self.dataChanged.emit(
                self.index(index.row(), 0), self.index(index.row(), 3)
            )
            return False
        self.error_key = None
        self.dataChanged.emit(self.index(index.row(), 0), self.index(index.row(), 3))
        self.edited.emit(p)
        return True


class ParamDelegate(QStyledItemDelegate):
    def createEditor(self, parent, option, index):
        p = index.model().param(index.row())
        if p.type == "enum":
            editor = QComboBox(parent)
            choices = list(p.choices)
            if p.value not in choices:
                choices.append(p.value)
            editor.addItems([str(c) for c in choices])
            return editor
        if p.type == "int":
            editor = QSpinBox(parent)
            lo = int(p.minimum) if p.minimum is not None else -(2**31) + 1
            hi = int(p.maximum) if p.maximum is not None else 2**31 - 1
            editor.setRange(lo, hi)
            editor.setKeyboardTracking(False)
            return editor
        if p.type == "float":
            editor = ScientificSpinBox(parent)
            editor.setRange(
                p.minimum if p.minimum is not None else -1e300,
                p.maximum if p.maximum is not None else 1e300,
            )
            return editor
        editor = QLineEdit(parent)
        if p.type in ("list", "matrix"):
            editor.setPlaceholderText(
                "[a, b, c]" if p.type == "list" else "[[a, b], [c, d]]"
            )
        return editor

    def setEditorData(self, editor, index):
        p = index.model().param(index.row())
        if isinstance(editor, QComboBox):
            editor.setCurrentText(str(p.value))
        elif isinstance(editor, QSpinBox):
            editor.setValue(int(p.value))
        elif isinstance(editor, QDoubleSpinBox):
            editor.setValue(float(p.value))
            editor.lineEdit().setText(_format_float(p.value))
        else:
            editor.setText(p.text())
            editor.selectAll()

    def setModelData(self, editor, model, index):
        if isinstance(editor, QComboBox):
            model.setData(index, editor.currentText())
        elif isinstance(editor, QSpinBox):
            editor.interpretText()
            model.setData(index, editor.value())
        elif isinstance(editor, QDoubleSpinBox):
            text = editor.lineEdit().text().strip()
            model.setData(index, text if text else editor.value())
        else:
            model.setData(index, editor.text())

    def updateEditorGeometry(self, editor, option, index):
        editor.setGeometry(option.rect.adjusted(0, -2, 0, 2))


class _ShortcutGuard(QObject):
    """Application event filter giving the editor Ctrl+S / Ctrl+F while focus is inside it.

    The main window may bind the same keys as window shortcuts, which Qt would
    otherwise report as ambiguous. The guard is a child of the editor, so Qt
    drops it from the application's filter list when the editor is destroyed.
    """

    def __init__(self, editor):
        super().__init__(editor)
        self.editor = editor
        QApplication.instance().installEventFilter(self)

    def eventFilter(self, obj, event):
        if event.type() not in (
            QEvent.ShortcutOverride,
            QEvent.KeyPress,
        ) or not isinstance(obj, QWidget):
            return False
        editor = self.editor
        if not editor.isVisible() or not (obj is editor or editor.isAncestorOf(obj)):
            return False
        if event.matches(QKeySequence.Save):
            if event.type() == QEvent.KeyPress:
                editor.save()
        elif event.matches(QKeySequence.Find):
            if event.type() == QEvent.KeyPress:
                editor.focus_search()
        else:
            return False
        event.accept()
        return True


class ParamEditor(QWidget):
    changed = pyqtSignal()
    saved = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.pset = None
        self._scope = None  # None | ("file", ParamFile) | ("group", ParamFile, name)
        self._build()
        self._filter_timer = QTimer(self)
        self._filter_timer.setSingleShot(True)
        self._filter_timer.setInterval(120)
        self._filter_timer.timeout.connect(self._refresh)
        self._status_timer = QTimer(self)
        self._status_timer.setSingleShot(True)
        self._status_timer.timeout.connect(lambda: self.status.setText(""))
        self._guard = _ShortcutGuard(self)
        self._set_enabled()

    # -- construction -------------------------------------------------------
    def _build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        bar = QHBoxLayout()
        bar.setSpacing(8)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search name, description or value   (Ctrl+F)")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(lambda: self._filter_timer.start())
        bar.addWidget(self.search, 1)
        self.modified_only = QCheckBox("Modified only")
        self.modified_only.toggled.connect(self._refresh)
        bar.addWidget(self.modified_only)
        self.count = label("", 9, MUTED)
        bar.addWidget(self.count)
        bar.addSpacing(8)
        self.save_button = _button("Save", self.save, "download", primary=True)
        self.save_button.setToolTip(
            "Write edited values (Ctrl+S); comments and formatting are kept"
        )
        self.revert_button = _button("Revert", self.revert, "reset")
        self.revert_button.setToolTip("Discard unsaved edits and reload from disk")
        self.reset_button = _button(
            "Reset all", lambda: self.reset_all(confirm=True), "reset"
        )
        self.reset_button.setToolTip("Set every parameter back to its default")
        self.folder_button = _button("", self.open_folder, "folder")
        self.folder_button.setToolTip("Open the configuration folder")
        self.folder_button.setFixedWidth(40)
        for widget in (
            self.save_button,
            self.revert_button,
            self.reset_button,
            self.folder_button,
        ):
            bar.addWidget(widget)
        layout.addLayout(bar)
        split = QSplitter(Qt.Horizontal)
        split.setHandleWidth(12)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Group", ""])
        self.tree.setColumnCount(2)
        self.tree.header().setStretchLastSection(False)
        self.tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        self.tree.header().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.tree.setMinimumWidth(220)
        self.tree.itemSelectionChanged.connect(self._scope_changed)
        split.addWidget(self.tree)
        self.model = ParamTableModel(self)
        self.model.rejected.connect(lambda text: self.notify(text, error=True))
        self.model.edited.connect(self._edited)
        self.table = QTableView()
        self.table.setStyleSheet(_TABLE_STYLE)
        self.table.setModel(self.model)
        self.table.setItemDelegateForColumn(1, ParamDelegate(self.table))
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(
            QAbstractItemView.DoubleClicked
            | QAbstractItemView.EditKeyPressed
            | QAbstractItemView.SelectedClicked
        )
        self.table.verticalHeader().hide()
        self.table.verticalHeader().setDefaultSectionSize(30)
        self.table.setWordWrap(False)
        self.table.setTextElideMode(Qt.ElideRight)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Interactive)
        header.setSectionResizeMode(1, QHeaderView.Interactive)
        header.setSectionResizeMode(2, QHeaderView.Interactive)
        header.setSectionResizeMode(3, QHeaderView.Stretch)
        header.resizeSection(0, 270)
        header.resizeSection(1, 230)
        header.resizeSection(2, 170)
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._context_menu)
        split.addWidget(self.table)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([240, 900])
        layout.addWidget(split, 1)
        self.status = label("", 9, MUTED)
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

    # -- loading ----------------------------------------------------------------
    def set_param_set(self, pset):
        self.pset = pset
        self._scope = None
        self._rebuild_tree()
        self._refresh()
        self._set_enabled()

    def load_files(self, paths, reference=None):
        self.set_param_set(ParamSet.load([Path(p) for p in paths], reference=reference))

    def load_estimator_config(self, path, reference=None):
        self.set_param_set(ParamSet.from_estimator_config(path, reference=reference))

    @property
    def param_set(self):
        return self.pset

    def has_unsaved(self):
        return bool(self.pset and self.pset.has_unsaved())

    # -- actions ------------------------------------------------------------------
    def save(self):
        if not self.pset or not self.pset.has_unsaved():
            return []
        try:
            paths = self.pset.save()
        except OSError as error:
            self.notify(f"Could not save: {error}", error=True)
            return []
        self._after_reload()
        for path in paths:
            self.saved.emit(str(path))
        self.notify("Saved " + ", ".join(Path(p).name for p in paths))
        return paths

    def revert(self):
        if not self.pset:
            return
        try:
            self.pset.revert()
        except (OSError, ValueError) as error:
            self.notify(f"Could not reload: {error}", error=True)
            return
        self._after_reload()
        self.changed.emit()
        self.notify("Reloaded from disk")

    def reset_all(self, confirm=True):
        if not self.pset:
            return
        count = len(self.pset.modified)
        if not count:
            return
        if confirm:
            answer = QMessageBox.question(
                self,
                "Reset all parameters",
                f"Reset {count} modified parameters to their defaults?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return
        self.pset.reset_all()
        self._after_edit()
        self.notify(f"{count} parameters reset to defaults (unsaved)")

    def reset(self, key):
        self.pset.reset(key)
        self._after_edit()

    def open_folder(self):
        if self.pset and self.pset.files and self.pset.files[0].path:
            QDesktopServices.openUrl(
                QUrl.fromLocalFile(str(self.pset.files[0].path.parent))
            )

    def focus_search(self):
        self.search.setFocus()
        self.search.selectAll()

    def notify(self, text, error=False):
        self.status.setText(text)
        self.status.setStyleSheet(
            f"color:{RED if error else TEAL}; background:transparent; border:none;"
        )
        self._status_timer.start(9000 if error else 5000)

    # -- internals ----------------------------------------------------------------
    def _edited(self, param):
        self._update_tree_counts()
        self._update_count()
        self._set_enabled()
        self.changed.emit()

    def _after_edit(self):
        self._refresh()
        self._edited(None)

    def _after_reload(self):
        self.pset.reindex()
        self._rebuild_tree()
        self._refresh()
        self._set_enabled()

    def _set_enabled(self):
        loaded = self.pset is not None
        unsaved = self.has_unsaved()
        self.save_button.setEnabled(unsaved)
        self.revert_button.setEnabled(unsaved)
        self.reset_button.setEnabled(loaded and bool(self.pset.modified))
        self.folder_button.setEnabled(
            loaded and bool(self.pset.files) and self.pset.files[0].path is not None
        )
        self.search.setEnabled(loaded)

    def _rebuild_tree(self):
        selected = self._scope_key(self._scope)
        self.tree.blockSignals(True)
        self.tree.clear()
        if not self.pset:
            self.tree.blockSignals(False)
            return
        root = QTreeWidgetItem(["All parameters", ""])
        root.setData(0, Qt.UserRole, None)
        self.tree.addTopLevelItem(root)
        for f in self.pset.files:
            item = QTreeWidgetItem([f.name, ""])
            item.setData(0, Qt.UserRole, ("file", f))
            item.setToolTip(0, str(f.path) if f.path else f.name)
            self.tree.addTopLevelItem(item)
            groups = []
            for p in f.params:
                if p.group not in groups:
                    groups.append(p.group)
            for g in groups:
                child = QTreeWidgetItem([g, ""])
                child.setData(0, Qt.UserRole, ("group", f, g))
                item.addChild(child)
            item.setExpanded(True)
        self._update_tree_counts()
        target = None
        iterator = [
            self.tree.topLevelItem(i) for i in range(self.tree.topLevelItemCount())
        ]
        while iterator:
            item = iterator.pop(0)
            if self._scope_key(item.data(0, Qt.UserRole)) == selected:
                target = item
                break
            iterator.extend(item.child(i) for i in range(item.childCount()))
        (target or root).setSelected(True)
        self._scope = (target or root).data(0, Qt.UserRole)
        self.tree.blockSignals(False)

    @staticmethod
    def _scope_key(scope):
        if scope is None:
            return None
        if scope[0] == "file":
            return ("file", str(scope[1].path))
        return ("group", str(scope[1].path), scope[2])

    def _scope_params(self, scope):
        if scope is None:
            return self.pset.params
        if scope[0] == "file":
            return scope[1].params
        return [p for p in scope[1].params if p.group == scope[2]]

    def _update_tree_counts(self):
        if not self.pset:
            return
        items = [
            self.tree.topLevelItem(i) for i in range(self.tree.topLevelItemCount())
        ]
        while items:
            item = items.pop(0)
            params = self._scope_params(item.data(0, Qt.UserRole))
            modified = sum(1 for p in params if p.modified)
            unsaved = any(p.dirty for p in params) or (
                item.data(0, Qt.UserRole) is not None
                and item.data(0, Qt.UserRole)[0] == "file"
                and item.data(0, Qt.UserRole)[1].has_unsaved()
            )
            item.setText(
                1, f"{modified}/{len(params)}" if modified else str(len(params))
            )
            item.setForeground(1, QColor(AMBER if modified else MUTED))
            f = item.font(0)
            f.setItalic(unsaved)
            item.setFont(0, f)
            items.extend(item.child(i) for i in range(item.childCount()))

    def _scope_changed(self):
        selected = self.tree.selectedItems()
        self._scope = selected[0].data(0, Qt.UserRole) if selected else None
        self._refresh()

    def _matches(self, key, p, needle):
        if not needle:
            return True
        haystack = " ".join((key, p.description, p.notes, p.text(), p.group)).lower()
        return all(term in haystack for term in needle.lower().split())

    def _refresh(self):
        if not self.pset:
            self.model.set_rows(None, [])
            self._update_count()
            return
        current = self.table.currentIndex()
        current_key = (
            self.model.rows[current.row()][0]
            if current.isValid() and current.row() < len(self.model.rows)
            else None
        )
        scope = set(id(p) for p in self._scope_params(self._scope))
        needle = self.search.text().strip()
        only = self.modified_only.isChecked()
        rows = [
            (key, p)
            for key, p in self.pset.items()
            if id(p) in scope
            and (not only or p.modified or p.dirty)
            and self._matches(key, p, needle)
        ]
        self.model.set_rows(self.pset, rows)
        if current_key is not None:
            for i, (key, _) in enumerate(rows):
                if key == current_key:
                    self.table.selectRow(i)
                    break
        self._update_count()

    def _update_count(self):
        if not self.pset:
            self.count.setText("")
            return
        total = len(self.pset.params)
        modified = len(self.pset.modified)
        unsaved = len(self.pset.dirty)
        parts = [f"{len(self.model.rows)} of {total}"]
        if modified:
            parts.append(f"{modified} modified")
        if unsaved:
            parts.append(f"{unsaved} unsaved")
        self.count.setText(" · ".join(parts))

    def _context_menu(self, position):
        index = self.table.indexAt(position)
        if not index.isValid():
            return
        key, p = self.model.rows[index.row()]
        menu = QMenu(self)
        reset = menu.addAction("Reset to default")
        reset.setEnabled(p.modified and not p.readonly)
        menu.addSeparator()
        copy_name = menu.addAction("Copy name")
        copy_value = menu.addAction("Copy value")
        action = menu.exec_(self.table.viewport().mapToGlobal(position))
        if action is reset:
            self.reset(p)
        elif action is copy_name:
            QApplication.clipboard().setText(p.key)
        elif action is copy_value:
            QApplication.clipboard().setText(p.text())


__all__ = ["ParamEditor", "ParamTableModel", "ParamDelegate", "ScientificSpinBox"]
