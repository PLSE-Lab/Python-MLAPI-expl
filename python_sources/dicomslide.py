"""

   Support for DICOM WSI files - RAM optimized. 

   To create a DICOM WSI file out of an openslide-supported WSI file, use wsi2dcm, available from:

   https://github.com/GoogleCloudPlatform/wsi-to-dicom-converter
   
   
   Note: This approach uses a handfull of black magic, since the DCM files do not have an index for the tiles.
         Thus, the approach is the following:
             1. Use pydicom to read the image, however keep it from loading the complete pixel data for the tileset 
                (takes up gigabytes of RAM)
             2. Backup the PixelData object of the pydicom.Dataset (which is an empty pydicom.dataelem.RawDataElement)
             3. Look for all frame offsets (DicomSlide._find_frame_offsets) for later use.
                Since - at least in all my WSI generated through wsi2dcm - no frame offsets were provided by the file,
                we need to read the complete beast initially. The PixelData object is used for this, it is populated 
                upon first use.
             4. Restore the PixelData object with the backup made earlier.
             
             Together with the object offset of the original PixelData object, we now have all offsets required to
             read the individual tiles with PIL.

"""

import numpy as np
import pydicom
from pydicom.filebase import DicomBytesIO
from pydicom.encaps import decode_data_sequence, get_frame_offsets, read_item
from pydicom.filereader import data_element_offset_to_value
from PIL import Image
import io
import sys
import os
from pydicom.tag import Tag

class DicomSlide():
    def _find_frame_offsets(self):
        
        data = self._dsstore[0].PixelData ## This will populate the PixelData array
        
        # Convert data into a memory-mapped file
        with DicomBytesIO(data) as fp:

            fp.is_little_endian = True
            BasicOffsetTable = read_item(fp)  # NOQA
            offsets = [fp.tell()]
            cnt=0
            while True:

                item = read_item(fp)
                cnt+=1
                
                # None is returned if get to Sequence Delimiter
                if not item:
                    break
                offsets.append(fp.tell())

            return offsets        
    
    def __init__(self, filename):

        self._ds = pydicom.dcmread(filename, defer_size=100e3)
        self._sequenceInstanceUID = self._ds.SeriesInstanceUID
        
        self._path, _ = os.path.split(filename)
        if (self._path==""):
            self._path='.'+os.sep

        self._dsstore = dict()

        self._dsstore[0] = self._ds

        self.levels = sorted(list(self._dsstore.keys()))
        self.geometry_imsize = [(self._dsstore[k][0x48,0x6].value,self._dsstore[k][0x48,0x7].value) for k in self.levels]
        self.geometry_tilesize = [(self._dsstore[k].Columns, self._dsstore[k].Rows) for k in self.levels]
        self.geometry_columns = [round(0.5+(self.geometry_imsize[k][0]/self.geometry_tilesize[k][0])) for k in self.levels]
        self.geometry_rows = [round(0.5 + (self.geometry_imsize[k][1] / self.geometry_tilesize[k][1] )) for k in self.levels]
        self.channels = self._ds[0x0028, 0x0002].value
        self.mpp_x = float(self._dsstore[0].SharedFunctionalGroupsSequence[0][0x028,0x9110][0][0x028,0x030][0])*1000
        self.mpp_y = float(self._dsstore[0].SharedFunctionalGroupsSequence[0][0x028,0x9110][0][0x028,0x030][1])*1000
        # store original value of PixelData to later defer
        self._pixeldata_deferred = self._ds._dict[Tag(0x7fe0,0x010)]        
        self._frame_offsets = self._find_frame_offsets()
        del self._ds._dict[Tag(0x7fe0,0x010)]
        self._ds._dict[Tag(0x7fe0,0x010)] = self._pixeldata_deferred

    @property
    def seriesInstanceUID(self) -> str:
        return self._sequenceInstanceUID

    @property
    def level_downsamples(self):
        return [self._dsstore[0].TotalPixelMatrixColumns/self._dsstore[k].TotalPixelMatrixColumns for k in self.levels]        

    @property 
    def level_dimensions(self):
        return [(self._dsstore[k].TotalPixelMatrixColumns,self._dsstore[k].TotalPixelMatrixRows) for k in self.levels]    

    @property
    def dimensions(self):
        return self.level_dimensions[0]

    def get_best_level_for_downsample(self,downsample):
        return np.argmin(np.abs(np.asarray(self.level_downsamples)-downsample))

    @property
    def level_count(self):
        return len(self.levels)

    def imagePos_to_id(self, imagePos:tuple, level:int):
        id_x, id_y = imagePos
        if (id_y>=self.geometry_rows[level]):
            id_x=self.geometry_columns[level] # out of range

        if (id_x>=self.geometry_columns[level]):
            id_y=self.geometry_rows[level] # out of range
        return (id_x+(id_y*self.geometry_columns[level]))
    
    def get_tile_randomaccess(self, pos, level):
        fp = self._ds.fileobj_type(self._ds.filename,'rb')
        offset = data_element_offset_to_value(self._pixeldata_deferred.is_implicit_VR,self._pixeldata_deferred.VR)
        position = self._pixeldata_deferred.value_tell + self._frame_offsets[pos]+8 # 8 bytes for empty offset index
        fp.seek(position)
        bytesread = fp.read(self._frame_offsets[pos+1]-self._frame_offsets[pos])
        if pos < self._dsstore[level].NumberOfFrames:
            return np.array(Image.open(io.BytesIO(bytesread)))
        else:
            return np.zeros((*self.geometry_tilesize[level], self.channels))

    def get_tile(self, pos, level:int):
        if pos < self._dsstore[level].NumberOfFrames:
            print('Read',len(bytesread),'Bytes:',bytesread[0:4])

            return np.array(Image.open(io.BytesIO(self._dsequence[level][pos])))
        else:
            return np.zeros((*self.geometry_tilesize[level], self.channels))

    def get_id(self, pixelX:int, pixelY:int, level:int) -> (int, int, int):

        id_x = round(-0.5+(pixelX/self.geometry_tilesize[level][1]))
        id_y = round(-0.5+(pixelY/self.geometry_tilesize[level][0]))
        
        return (id_x,id_y), pixelX-(id_x*self.geometry_tilesize[level][0]), pixelY-(id_y*self.geometry_tilesize[level][1]),

        
    def read_region(self, location: tuple, level:int, size:tuple):
        
        assert(level==0) # no support for higher levels, yet
        
        # convert to overall coordinates, if not in level 0
        if (self.level_downsamples[level]>1):
            location = [int(x/self.level_downsamples[level]) for x in location]

        lu, lu_xo, lu_yo = self.get_id(*list(location),level=level)
        rl, rl_xo, rl_yo = self.get_id(*[sum(x) for x in zip(location,size)], level=level)
        # generate big image
        bigimg = 255*np.ones(((rl[1]-lu[1]+1)*self.geometry_tilesize[level][0], (rl[0]-lu[0]+1)*self.geometry_tilesize[level][1], self.channels+1), np.uint8)
        for xi, xgridc in enumerate(range(lu[0],rl[0]+1)):
            for yi, ygridc in enumerate(range(lu[1],rl[1]+1)):
                if (xgridc<0) or (ygridc<0):
                    continue
                bigimg[yi*self.geometry_tilesize[level][0]:(yi+1)*self.geometry_tilesize[level][0],
                       xi*self.geometry_tilesize[level][1]:(xi+1)*self.geometry_tilesize[level][1],0:3] = \
                       self.get_tile_randomaccess(self.imagePos_to_id((xgridc,ygridc),level=level), level)
        # crop big image
        return Image.fromarray(bigimg[lu_yo:lu_yo+size[1],lu_xo:lu_xo+size[0]])
    

