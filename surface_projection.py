#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy
import sys
import skimage
import skimage.morphology
import skimage.transform
import skimage.filters
from tifffile import TiffFile, TiffWriter

import numpy
import scipy
import pathlib
import re

def grabIJTags(info):
    tags = {}
    ij_pat = re.compile("\\* (\\w+): (.*)$")
    digit = re.compile("^\\d+$")
    for line in info.split("\n"):
        m = ij_pat.match(line)
        if m:
            value = m.group(2)
            if digit.match(value):
                value = int(value)
            tags[m.group(1)] = value
    return tags

def loadImage(imageFile):
    imageFile = str(imageFile)
    print("loading %s"%imageFile)
    try:
        with TiffFile(imageFile) as tiff:
            data = []
            for p in tiff.pages:
                data.append(p.asarray())
            data = numpy.array(data)
            print("starting shape: %s"%(data.shape,))
            tags = tiff.imagej_metadata;
            if len(data.shape)==3:
                data = data.reshape((1, 1, *data.shape[:]))
            elif len(data.shape)==4:
                data = data.reshape( (data.shape[0], 1, *data.shape[1:] ) )
            elif len(data.shape)==5:
                data = numpy.rollaxis(data, 1, 0)
            
            frames = tags.get("frames", data.shape[0])
            channels = tags.get("channels", data.shape[1])
            slices = tags.get("slices", None)
            print("%s, %s, %s frames, channels, slices"%(frames, channels, slices))
            if slices is None:
                images = data.shape[0]*data.shape[1]*data.shape[2]
                slices = images//channels//frames
            
            data = data.reshape((frames, channels, slices, data.shape[3], data.shape[4]))
            print("final shape: %s"%(data.shape, ) )
            return data, tags
    except:
        raise Exception("Unable to load image: %s"%imageFile )
        
def saveImage(file_name, data):
    print("saving: ", data.shape)
    with TiffWriter(file_name, imagej=True) as writer:
        writer.save(data)


def createHeightMap(img, surface_blur=40, height_blur=5):
    """
       This will create a height map by blurring the image in 2d then finding the maximum
       along the z-axis. 
       
       If height_blur is included a float image will be returned with height values smoothed.
    
        
    """
    
    height_maps = []
    for stack in img:
    
        blurred = skimage.filters.gaussian(stack, sigma=[0, 0, surface_blur, surface_blur])
    
    
        #find the max in z direction.
        height_map = numpy.argmax(blurred, axis=1)*1.0
        print(height_map.shape)
        if height_blur>0:
            height_map = skimage.filters.gaussian(height_map, sigma=[0, height_blur, height_blur])
        #single channel, single z-slice
        height_maps.append([height_map])
        
    return numpy.array(height_maps, dtype="float32")

def projectAverage(mask, image, width = 2):
    """
        
        Uses the provided mask, assuming 1px with projection mask, and
        expands the mask, then sums along the z axis.
        
    """
    mask2 = scipy.ndimage.convolve1d(mask, numpy.ones((2*width + 1,), dtype="int8"), axis=0,mode="constant", cval=0, origin=0)
    count=numpy.sum(mask2, axis=0)
    
    return numpy.sum(mask2*image, axis=0)/count
    
def projectMax(mask, image, width=2):
    """
        Expands the provided mask along the z-axis and finds the maximum
        value along that with of the axis.
    """
    mask2 = scipy.ndimage.convolve1d(mask, numpy.ones((2*width + 1,), dtype="int8"), axis=0,mode="constant", cval=0, origin=0)
    return numpy.max(mask2*image, axis=0)



def createProjection( height_map, image, reduction=projectMax):
    """
        Uses the provided height_map to create a surface projection.
    """
    projections = []
    for map_stack, vol_stack in zip(height_map, image):
        hm = map_stack[0,0] #single channel/single slice
        vol = vol_stack[0] #first channel
        I, J = numpy.indices(hm.shape)
        if numpy.issubdtype(hm.dtype, numpy.integer):
            mask = numpy.zeros(image.shape)
            mask[hm, I, J] = 1
            proj = reduction(mask, image)
        else:
            mask_low = numpy.zeros(vol.shape)
            mask_high = numpy.zeros(vol.shape)
            height_map_low = numpy.array(hm, dtype="int")
            height_map_high = numpy.array( hm + 1, dtype="int")
            numpy.clip(height_map_high, 0, vol.shape[0]-1, height_map_high)
            fractional = hm - height_map_low
            
            mask_low[height_map_low, I, J] = 1 
            mask_high[height_map_high, I, J] = 1
            
            a = reduction(mask_low, vol)
            b = reduction(mask_high, vol)
            
            proj = ( 1 - fractional ) * a + fractional*b
        print("projection: ", proj.shape)
        projections.append([[ proj ]]) #1 channel 1 slice
    return projections

import click

@click.group()
def projections():
    pass

@projections.command("p")
@click.argument("height_map_file", type=click.Path(exists=True))
@click.argument("to_project", nargs=-1, type=click.Path(exists=True))
def projectOnto(height_map_file, to_project):
    height_map, tags = loadImage(height_map_file)
    
    for image in to_project:
        img, tags = loadImage(image)
        proj = createProjection(height_map, img, projectAverage)
        proj = numpy.array(proj, dtype="uint16")
        projection_name = image.replace(".tif", "") + "-proj.tif"
        saveImage(projection_name, proj)
        
@projections.command("h")
@click.argument("reference", type=click.Path(exists=True))
@click.option("-s", "--surface_blur", type=click.INT, default=40)
@click.option("-b", "--height_blur", type=click.INT, default=5)
@click.option("-h", "--height_map_file", type=click.Path())
@click.option("-o", "--output_folder", type=click.Path(exists=True), default=".")
def heightMap(reference, surface_blur, height_blur,  height_map_file, output_folder):
    if height_map_file:
        hm_name = height_map_file
    else:
        hm_name = reference.replace(".tif", "") + "-hm.tif"
    hm_path = pathlib.Path(output_folder, pathlib.Path(hm_name).name)

    projector, tags = loadImage(reference)
    height_map = createHeightMap(projector, surface_blur=surface_blur, height_blur=height_blur)

    saveImage(hm_name, height_map)
    
    
#    if project_reference:
#        proj = createProjection(height_map, projector, reduction=projectMax)
#        print("float values", numpy.max(proj), numpy.min(proj))
#        proj = numpy.array(proj, dtype="uint16")
#        print("int values", numpy.max(proj), numpy.min(proj))
#        skimage.io.imsave(reference.replace(".tif", "") + "-proj.tif", proj)
#
#    for image in to_project:
#        img = skimage.io.imread(image)
#        proj = createProjections(height_map, img, projectMax)
#        print("float values", numpy.max(proj), numpy.min(proj))
#        proj = numpy.array(proj, dtype="uint16")
#        print("int values", numpy.max(proj), numpy.min(proj))
#        projection_name = image.replace(".tif", "") + "-proj.tif"
#        skimage.io.imsave(proj, projection_name)


if __name__=="__main__":
    projections()
