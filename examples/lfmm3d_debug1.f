      implicit none
      integer ns,nt
      double precision, allocatable :: source(:,:)
      double precision, allocatable :: targ(:,:)
      double precision, allocatable :: charge(:)
      double precision, allocatable :: pot(:),potex(:)
      double precision, allocatable :: grad(:,:),gradex(:,:)

      double precision eps
      integer i,j,k,ntest
      double precision hkrand,thet,pi,erra,ra,thresh
      double precision tmp1,tmp2,tmp3,errg,rg
      

      pi = atan(1.0d0)*4.0d0

c
cc      initialize printing routine
c
      call prini(6,13)
      write(*,*)
      write(*,*)
      write(*,*) "================================="
      call prin2("This code is an example fortran driver*",i,0)
      call prin2("On output, the code prints sample pot,grad*",i,0)
      write(*,*)
      write(*,*)


      ns = 216
      
      nt = 216
      

      allocate(source(3,ns))
      allocate(targ(3,nt))
      allocate(charge(ns))
      allocate(pot(nt))
      allocate(grad(3,nt))


      eps = 1.0d-6

      write(*,*) "=========================================="

c
c   
c       example demonstrating use of 
c        source to source, charges, pot + gradient
c



c
cc      generate sources uniformly in the unit cube 
c
c
      open(unit=33,file='examples/srcloc.txt')
      do i=1,ns
        read(33,*) tmp1,tmp2,tmp3
        source(1,i) = tmp1
        source(2,i) = tmp2
        source(3,i) = tmp3
        charge(i) = hkrand(0)
      enddo
      close(33)

      open(unit=33,file='examples/targloc.txt')
      do i=1,nt
        read(33,*) tmp1,tmp2,tmp3
        targ(1,i) = tmp1
        targ(2,i) = tmp2
        targ(3,i) = tmp3
        pot(i) = 0
        grad(1,i) = 0
        grad(2,i) = 0
        grad(3,i) = 0 
      enddo
      close(33)

      call prin2('targ=*',targ,12)


      call lfmm3d_t_c_g(eps,ns,source,charge,
     1      nt,targ,pot,grad)
      call prin2("pot at sources=*",pot,12)
cc      call prin2("grad at sources=*",grad,3*nt)

c
c      test accuracy
c

      ntest = nt
      allocate(potex(ntest),gradex(3,ntest))
      thresh = 1.0d-16
      do i=1,ntest
        potex(i) = 0
        gradex(1,i) = 0
        gradex(2,i) = 0
        gradex(3,i) = 0
      enddo
      call l3ddirectcg(1,source,charge,ns,targ,ntest,
     1       potex,gradex,thresh)


      erra = 0
      ra = 0
      errg = 0
      rg = 0
      do i=1,ntest
         ra = ra + abs(potex(i))**2
         erra = erra + abs(pot(i)-potex(i))**2
         rg = rg + gradex(1,i)**2 + gradex(2,i)**2 + gradex(3,i)**2
         errg = errg + (gradex(1,i)-grad(1,i))**2
         errg = errg + (gradex(2,i)-grad(2,i))**2
         errg = errg + (gradex(3,i)-grad(3,i))**2
      enddo

      call prin2('potex=*',potex,ntest)
      call prin2('pot=*',pot,ntest)
      call prin2('gradex=*',gradex,3*ntest)
      call prin2('grad=*',grad,3*ntest)

      erra = sqrt(erra/ra)
      errg = sqrt(errg/rg)

      call prin2('error=*',erra,1)
      call prin2('error gradient=*',errg,1)

      stop
      end
c----------------------------------------------------------
c
cc
c
c
