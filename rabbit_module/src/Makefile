# CXXFLAGS += -Wall -Wextra -Wcast-align -Wcast-qual -Wconversion -Wfloat-equal \
# 	    -Wformat=2 -Winit-self -Wmissing-declarations \
# 	    -Wmissing-include-dirs -Wpointer-arith -Wredundant-decls \
# 	    -Wswitch-default -Wuninitialized -Wwrite-strings \
# 	    -Wno-sign-conversion -Wno-unused-function \
#             -Wno-missing-declarations \
#             -fopenmp -std=c++14 -mcx16 -O3 -DNDEBUG 

FLAGS =  -fopenmp -std=c++14 -mcx16 -O3
LDLIBS   += -ltcmalloc_minimal -lnuma
TARGETS   = reorder

$(TARGETS): %: %.cc
	$(LINK.cc) -MD $(FLAGS) -o $@ $< $(LDLIBS)
	@cp $*.d .$*.P; \
	  sed -e 's/#.*//' -e 's/^[^:]*: *//' -e 's/ *\\$$//' \
	      -e '/^$$/ d' -e 's/$$/ :/' < $*.d >> .$*.P; \
	  rm -f $*.d

.PHONY: clean
clean:
	$(RM) $(TARGETS) $(TARGETS:%=.%.P)

-include .*.P

