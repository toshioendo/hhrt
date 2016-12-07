ARCDIR = lib
#ARCDIR = noplib

include make.inc

LDFLAGS += -lrt
CFLAGS += -I./$(ARCDIR)

LIBS = $(ARCDIR)/libhhrt.a 
#libipsm/libipsm.so

all: 7pstencil

7pstencil: 7pstencil.o libhhrt Makefile
	$(LD) $@.o -o $@ $(LIBS) $(LDFLAGS)

pingpong: pingpong.o $(LIBS) Makefile
	$(LD) $@.o -o $@ $(LDFLAGS)

libhhrt:
	cd $(ARCDIR); make ; cd ..

libipsm/libipsm.so: 
	cd libipsm; make ; cd ..

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
