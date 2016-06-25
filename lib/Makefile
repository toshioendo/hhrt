include ../make.inc

ARCOBJS = hhmain.o hhsched.o hhmem.o hhswapper.o hhheap.o hhmpi.o hhcuda.o hhshm.o hhaux.o ipsm.o
ARC = libhhrt.a
HDRS = hhrt.h hhrt_impl.h hhrt_common.h

all: $(ARC) hhview

$(ARC): $(ARCOBJS)
	ar r $@ $(ARCOBJS)

hhview: hhview.o ipsm.o
	$(LD) $@.o ipsm.o -o $@ $(LDFLAGS)

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
