HHRTROOT = ..
HHRTLIBDIR = $(HHRTROOT)/lib

include $(HHRTROOT)/make.inc

CFLAGS += -I$(HHRTLIBDIR)

LIBS = $(HHRTLIBDIR)/libhhrt.a 
LDFLAGS += -lrt -L$(HHRTLIBDIR) -lhhrt

all: gcomm

gcomm: gcomm.o libhhrt Makefile
	$(LD) $@.o -o $@ $(LDFLAGS)

pingpong: pingpong.o $(LIBS) Makefile
	$(LD) $@.o -o $@ $(LDFLAGS)

libhhrt:
	pushd .; cd $(HHRTLIBDIR); make ; popd

%.o : %.c $(HDRS)
	$(CC) $(CFLAGS) -c $< -o $*.o

%.o : %.cc $(HDRS)
	$(CXX) $(CFLAGS) -c $< -o $*.o

%.o : %.cu $(HDRS)
	nvcc $(NVFLAGS) -c $< -o $*.o

clean:
	cd $(HHRTLIBDIR); make clean; cd ..
	rm -f *.o
	rm -f *~
	rm -f core
	rm -f core.*
	rm -f a.out
