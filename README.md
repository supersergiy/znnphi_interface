# Compilation instructions
To compile your network run:

`sudo docker run -it  -v /opt/intel/licenses -v $(pwd):/seungmount  seunglab/pznet:devel /opt/znnphi_interface/code/scripts/sergify.py -n /seungmount/deploy.prototxt -w /seungmount/weights.h5 -o /seungmount/out`

You'll need to change some parameters depending on where your files are stored(docker -v params and sergify.py -n -w and -o)

For more information on `sergify.py` you can run 

`sudo docker run -it  seunglab/pznet:devel /opt/znnphi_interface/code/scripts/sergify.py -h`
