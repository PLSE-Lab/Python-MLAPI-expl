from abc import ABC
from itertools import chain, islice
from math import ceil

from IPython.display import clear_output, display
from ipywidgets import Layout, widgets

from fastai.vision import *

__all__ = 'DatasetFormatter', 'PredictionsCorrector'

class DatasetFormatter():
    @staticmethod
    def padded_ds(ll_input, size=(250, 300), resize_method=ResizeMethod.CROP, padding_mode='zeros', **kwargs):
        return ll_input.transform(tfms=crop_pad(), size=size, resize_method=resize_method, padding_mode=padding_mode)

    @classmethod
    def from_most_unsure(cls, learn:Learner, num=50) -> Tuple[DataLoader, List[int], Sequence[str], List[str]]:
        preds, _ = learn.get_preds(DatasetType.Test)
        classes = learn.data.train_dl.classes
        labels = [classes[i] for i in preds.argmax(dim=1)]

        most_unsure = preds.topk(2, dim=1)[0] @ torch.tensor([1.0, -1.0])
        most_unsure.abs_()
        idxs = most_unsure.argsort()[:num].tolist()
        return cls.padded_ds(learn.data.test_dl), idxs, classes, labels

@dataclass
class ImgData:
    jpg_blob: bytes
    label: str
    payload: Mapping

class BasicImageWidget(ABC):
    def __init__(self, dataset:LabelLists, fns_idxs:Collection[int], batch_size=5, drop_batch_on_nonfile=False,
                 classes:Optional[Sequence[str]]=None,labels:Optional[Sequence[str]]=None,
                 on_next_batch:Optional[Callable[[Tuple[Mapping, ...]], Any]]=None):
        super().__init__()
        self._dataset,self.batch_size,self._labels,self.on_next_batch = dataset,batch_size,labels,on_next_batch
        self._classes = classes or dataset.classes
        self._all_images = self.create_image_list(fns_idxs, drop_batch_on_nonfile)

    @staticmethod
    def make_img_widget(img:bytes, layout=Layout(height='250px', width='300px'), format='jpg') -> widgets.Image:
        return widgets.Image(value=img, format=format, layout=layout)

    @staticmethod
    def make_button_widget(label:str, handler:Callable, batch_idx:Optional[int]=None,
                           style:str=None, layout=Layout(width='auto')) -> widgets.Button:
        btn = widgets.Button(description=label, layout=layout)
        btn.on_click(handler)
        if style is not None: btn.button_style = style
        if batch_idx is not None: btn.batch_idx = batch_idx
        return btn

    @staticmethod
    def make_dropdown_widget(options:Collection, value, handler:Callable, batch_idx:Optional[int]=None,
                             description='', layout=Layout(width='auto')) -> widgets.Dropdown:
        dd = widgets.Dropdown(description=description, options=options, value=value, layout=layout)
        dd.observe(handler, names='value')
        if batch_idx is not None: dd.batch_idx = batch_idx
        return dd

    @staticmethod
    def make_horizontal_box(children:Collection[widgets.Widget], layout=Layout()) -> widgets.HBox:
        return widgets.HBox(children, layout=layout)

    @staticmethod
    def make_vertical_box(children:Collection[widgets.Widget],
                          layout=Layout(width='auto', height='300px', overflow_x="hidden")) -> widgets.VBox:
        return widgets.VBox(children, layout=layout)

    def create_image_list(self, fns_idxs:Collection[int], drop_batch_on_nonfile=False) -> Iterator[ImgData]:
        items = self._dataset.x.items
        idxs = ((i for i in fns_idxs if items[i].is_file())
                if not drop_batch_on_nonfile
                else chain.from_iterable(c for c in chunks(fns_idxs, self.batch_size)
                                           if all(items[i].is_file() for i in c)))
        for i in idxs: yield ImgData(self._dataset.x[i]._repr_jpeg_(), self._get_label(i), self.make_payload(i))

    def _get_label(self, idx):
        return self._labels[idx] if self._labels is not None else self._classes[self._dataset.y[idx].data]

    @abstractmethod
    def make_payload(self, idx:int) -> Mapping: pass
    def _get_change_payload(self, change_owner): return self._batch_payloads[change_owner.batch_idx]

    def next_batch(self, _=None):
        if self.on_next_batch and hasattr(self, '_batch_payloads'): self.on_next_batch(self._batch_payloads)
        batch = tuple(islice(self._all_images, self.batch_size))
        self._batch_payloads = tuple(b.payload for b in batch)
        self.render(batch)

    @abstractmethod
    def render(self, batch:Tuple[ImgData]): pass

class PredictionsCorrector(BasicImageWidget):
    def __init__(self, dataset:LabelLists, fns_idxs:Collection[int],
                 classes:Sequence[str], labels:Sequence[str], batch_size:int=5):
        super().__init__(dataset, fns_idxs, batch_size, classes=classes, labels=labels)
        self.corrections:Dict[int, str] = {}
        self.next_batch()

    def show_corrections(self, ncols:int, **fig_kw):
        nrows = ceil(len(self.corrections) / ncols)
        fig, axs = plt.subplots(nrows, ncols, **fig_kw)
        axs, extra_axs = np.split(axs.flatten(), (len(self.corrections),))

        for (idx, new), ax in zip(sorted(self.corrections.items()), axs):
            old = self._get_label(idx)
            self._dataset.x[idx].show(ax=ax, title=f'{idx}: {old} -> {new}')

        for ax in extra_axs:
            ax.axis('off')
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)

    def corrected_labels(self) -> List[str]:
        corrected = list(self._labels)
        for i, l in self.corrections.items(): corrected[i] = l
        return corrected

    def make_payload(self, idx:int): return {'idx': idx}

    def render(self, batch:Tuple[ImgData]):
        clear_output()
        if not batch:
            return display('No images to show :)')
        else:
            display(self.make_horizontal_box(self.get_widgets(batch)))
            display(self.make_button_widget('Next Batch', handler=self.next_batch, style='primary'))

    def get_widgets(self, batch:Tuple[ImgData]):
        widgets = []
        for i, img in enumerate(batch):
            img_widget = self.make_img_widget(img.jpg_blob)
            dropdown = self.make_dropdown_widget(options=self._classes, value=img.label,
                                                 handler=self.relabel, batch_idx=i)
            widgets.append(self.make_vertical_box((img_widget, dropdown)))
        return widgets

    def relabel(self, change):
        self.corrections[self._get_change_payload(change.owner)['idx']] = change.new
