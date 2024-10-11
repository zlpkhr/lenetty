#!/bin/bash

# Let us not pollute too much our working directory, given the huge
# number of files we will produce
rd=results
mkdir -p $rd
rm -f $rd/labels
grep label float_images.c|cut -d' ' -f 3 >> $rd/labels
for fptype in float double fp16 bfloat16 fp8e5m2 fp8e4m3 ; do
	touch ./floatx-lenet.cpp
	make DEFFP=$fptype && mv ./floatx-lenet ./floatx-lenet-$fptype
	# Check all divisors to determine the best one(s)
	for i in {1..255} ; do
	{
		./floatx-lenet-$fptype $i | cut -d' ' -f 3 > $rd/result-$fptype-$i
		diff -U 0 $rd/labels $rd/result-$fptype-$i | grep -c ^@ > $rd/accuracy-$fptype-$i
	} &
	done
	wait
	(for i in {1..255}; do echo -n $i " "; cat $rd/result-$fptype-$i | diff -U 0 labels - | grep -c ^@; done) | sort -n -k 2 | head -20 > scale-$fptype
done
