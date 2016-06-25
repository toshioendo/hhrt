HHRTROOT = ..
ARCDIR = $(HHRTROOT)/lib
#ARCDIR = $(HHRTROOT)/noplib

include $(HHRTROOT)/make.inc

LDFLAGS += -lrt
CFLAGS += -I$(ARCDIR)

LIBS = $(ARCDIR)/libhhrt.a 

all: gcomm

gcomm: gcomm.o libhhrt Makefile
	$(LD) $@.o -o $@ $(LIBS) $(LDFLAGS)

pingpong: pingpong.o $(LIBS) Makefile
	$(LD) $@.o -o $@ $(LDFLAGS)

libhhrt:
	pushd .; cd $(ARCDIR); make ; popd

%.o : %.c $(HDRS)
	$(CC) $(CFLAGS) -c $< -o $*.o

%.o : %.cc $(HDRS)
	$(CXX) $(CFLAGS) -c $< -o $*.o

%.o : %.cu $(HDRS)
	nvcc $(NVFLAGS) -c $< -o $*.o

clean:
	cd $(ARCDIR); make clean; cd ..
	rm -f *.o
	rm -f *~
	rm -f core
	rm -f core.*
	rm -f $(APP)
	rm -f a.out
