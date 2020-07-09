# Invisibilty-Cloak

Using opencv masks, the module renders objects of a given color invisible to the camera.
The range finder module provided within can help identify the color range of any object.

## Usage
```commandline
python invisibility_cloak.py --filter red
python invisibility_cloak.py --filter custom --lower 10,150,0 --upper 140,255,255

python range_finder.py --filter RGB --image /path/image.png
python range_finder.py --filter HSV --webcam
```


