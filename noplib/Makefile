include ../make.inc

ARCOBJS = hhmain.o
ARC = libhhrt.a
HDRS = hhrt.h

all: $(ARC)

$(ARC): $(ARCOBJS)
	ar r $@ $(ARCOBJS)

%.o : %.c $(HDRS)
	$(CC) $(CFLAGS) -c $< -o $*.o

%.o : %.cc $(HDRS)
	$(CXX) $(CFLAGS) -c $< -o $*.o

%.o : %.cu $(HDRS)
	nvcc $(NVFLAGS) -c $< -o $*.o

clean:
	rm -f *.o
	rm -f *~
	rm -f core
	rm -f core.*
	rm -f $(ARC)
	rm -f a.out
