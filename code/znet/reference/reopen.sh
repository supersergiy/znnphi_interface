echo "Redoing everything"
HERE='/home/ubuntu/znnphi_interface/code/znet/reference'
rm -rf $HERE/data $HERE/nets
tar -xzf $HERE/bundle.tar.gz  -C $HERE
