from ngsolve import *
from netgen.csg import *
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from netgen.geom2d import SplineGeometry, CSG2d,Rectangle
from ngsolve.meshes import MakeStructured2DMesh, Make1DMesh, MakeStructured3DMesh
from math import isnan
import time
from ngsolve.nonlinearsolvers import Newton
from ngsolve.webgui import Draw
ngsglobals.msg_level = 0

class LogDensity:
    step  = 0
    order = 1
    log   = ""
    minVal = -40
    newtonCount = []
    newtonTolerance = 1e-12
    cutOff = 1e-14
    
    def __init__(self,tstart=0,dt=1e-2,
                 plot=True,plotEvery=10,
                 xrange=(-10,10),nx=1,
                 method= 'logDensity', m=3,s0=3,dim=1,
                 IC = 'BB', quads = False,dcFlag="",
                 printTime=True, plotInterface=True
                 ):
    
        self.tstart, self.dt =  Parameter(tstart), Parameter(dt)
        self.xmin,self.xmax = xrange
        self.nx,self.dim = nx,dim
        self.method = method
        self.m,self.s0 = m,s0
        self.k = 1/(self.m-1+2/self.dim)
        self.IC = IC
        self.plot, self.plotEvery = plot,plotEvery
        self.quads = quads
        self.dcFlag = dcFlag
        self.printTime = printTime
        self.plotInterface = plotInterface
        
        self.createMesh()
        self.createIR()
        self.createFESpace()
        self.TnT()
        self.createGF()
        self.setInitialData()
        self.createForms()
    
        if self.plot: self.createPlot()
            
    def createMesh(self):
        if self.dim==1:
            self.mesh = Make1DMesh(n = self.nx, mapping=lambda x:self.xmin+(self.xmax-self.xmin)*x)
        if self.dim==2:
            if self.quads: self.mesh = MakeStructured2DMesh(quads=self.quads, nx=self.nx, ny=self.nx, 
                                            mapping=lambda x,y:(self.xmin+(self.xmax-self.xmin)*x, 
                                                                           self.xmin+(self.xmax-self.xmin)*y))
            else:
                if self.IC != 'BB':
                    geo = CSG2d()
                    rect = Rectangle( pmin=(self.xmin,self.xmin), pmax=(self.xmax,self.xmax) )
                    geo.Add(rect)
                    self.mesh = Mesh(geo.GenerateMesh(maxh=(self.xmax-self.xmin)/self.nx))
                else:    
                    geo = SplineGeometry()
                    geo.AddCircle((0,0), self.xmax)
                    self.mesh = Mesh(geo.GenerateMesh(maxh=(self.xmax-self.xmin)/self.nx))
        if self.dim==3:
            if self.quads: self.mesh = MakeStructured3DMesh(hexes=self.quads, nx=self.nx, ny=self.nx, nz=self.nx,
                                            mapping=lambda x,y,z:(self.xmin+(self.xmax-self.xmin)*x, 
                                                                 self.xmin+(self.xmax-self.xmin)*y,
                                                                 self.xmin+(self.xmax-self.xmin)*z))
            else:
                geo = CSGeometry()
                sphere = Sphere(Pnt(0,0,0),self.xmax)
                geo.Add(sphere)
                self.mesh = Mesh(geo.GenerateMesh(maxh=(self.xmax-self.xmin)/self.nx))
                
                
    def createIR(self):
        if self.dim==1:
            self.ir = IntegrationRule(points = [(0,0), (1,0)], weights = [1/2, 1/2] )
            self.ELE = SEGM
        if self.dim ==2:
            if self.quads:
                self.ir  = IntegrationRule(points = [(0,0), (1,0), (0,1),(1,1)], weights = [1/4, 1/4, 1/4, 1/4] )
                self.ELE = QUAD
            else:
                self.ir = IntegrationRule(points = [(0,0), (1,0), (0,1)], weights = [1/6, 1/6, 1/6] )
                self.ELE = TRIG
        if self.dim == 3:
            if self.quads:
                self.ir = IntegrationRule(points = [(0,0,0),(1,0,0),(0,1,0),(1,1,0),
                                                    (0,0,1),(1,0,1), (0,1,1),(1,1,1),],
                                          weights = [1/8, 1/8, 1/8, 1/8,
                                                     1/8, 1/8, 1/8, 1/8] )
                self.ELE = HEX
            else:
                self.ir = IntegrationRule(points = [(0,0,0), (1,0,0), (0,1,0),(0,0,1)], 
                                          weights = [1/24, 1/24, 1/24, 1/24] )
                self.ELE = TET
        self.dx = dx(intrules={self.ELE:self.ir})
    
    
    def createFESpace(self):
        self.fes = H1(self.mesh,order=self.order,dirichlet=self.dcFlag)
        self.activeDofs = BitArray(self.fes.ndof)
        
    def TnT(self):
        self.u,self.v = self.fes.TnT()
        
    def createGF(self):
        self.uh = GridFunction(self.fes)
        # Used to store data of previous iteration
        self.uh0 = GridFunction(self.fes)
        self.rho0 = GridFunction(self.fes)
        # Used in Newton's iterations
        self.rhok = GridFunction(self.fes)
        self.rhouk = GridFunction(self.fes)
        
    def createForms(self):
        self.a = BilinearForm(self.fes)
        self.a += self.m*self.rho0**self.m*grad(self.u)*grad(self.v)*self.dx
        self.a += self.rhok*self.u*self.v/self.dt*self.dx

        self.f = LinearForm(self.fes)
        self.f += (self.rhouk-self.rhok+self.rho0)*self.v/self.dt*self.dx
        if self.IC == 'accuracy':
            self.g = self.rhoex.Diff(self.tstart)-(self.rhoex**self.m).Diff(x).Diff(x)
            self.f += self.g*self.v*self.dx
        
        
    def setInitialData(self):
        if   self.dim == 1: 
            r2 = x**2
            self.pnts_x = np.linspace(self.xmin,self.xmax,self.nx+1)  
        elif self.dim == 2: r2 = x**2+y**2
        elif self.dim == 3: r2 = x**2+y**2+z**2
        else: raise Exception("Not available.")
        
        if self.IC == 'BB':
            val = self.s0-self.k*(self.m-1)/(2*self.dim*self.m)*r2/(self.tstart+1)**(2*self.k/self.dim)
            self.rhoex = (self.tstart+1)**(-self.k)*(IfPos(val, val**(1/(self.m-1)), 0))
        
        elif self.IC == 'waitingTime' and self.dim==1:
            self.rhoex = IfPos((x-pi/2)*(x+pi/2), 0, ((self.m-1)/self.m*cos(x)**2)**(1/(self.m-1)) )
            self.tWait = 1/2/(self.m+1)
        elif self.IC == 'accuracy' and self.dim==1:
            self.rhoex = 1+0.5*sin(pi*self.tstart)*sin(pi*x)
        elif self.IC == 'complexSupport' and self.dim==2:
            r0 = (x**2+y**2)**0.5
            r1 = (x**2+(y-0.75)**2)**0.5
            r2 = ((x-0.75)**2+y**2)**0.5
            dom1 = IfPos((r0-0.5)*(r0-1),0, 1)*(1-IfPos(x, 1,0)*IfPos(y,1,0))
            val1 = 625*(0.25**2-(r0-0.75)**2)**3
            dom2 = IfPos(r1-0.25, 0, 1)*IfPos(x, 1,0)
            val2 = 625*(0.25**2-r1**2)**3
            dom3 = IfPos(r2-0.25, 0, 1)*IfPos(y, 1,0)
            val3 =625*(0.25**2-r2**2)**3 
            self.rhoex = dom1*(IfPos(val1, val1, -val1))**(0.5/(self.m-1)) \
                + dom2*(IfPos(val2, val2, -val2))**(0.5/(self.m-1)) \
                + dom3*(IfPos(val3, val3, -val3))**(0.5/(self.m-1)) 
        elif self.IC == 'mergingGaussians' and self.dim==2:
            self.rhoex = exp(-20*((x-0.3)**2 + (y-0.3)**2))+exp(-20*((x+0.3)**2 + (y+0.3)**2))
        else: raise Exception("Not available.")
            
        
        self.rho0.vec[:] = 0
        self.rhok.vec[:] = 0
        
        self.rho0.Set(self.rhoex)
        if self.IC == 'waitingTime':
            self.uh0.Set(log(self.rhoex))
        else:
            self.uh0.vec.data = np.log(self.rho0.vec)

        
        self.pos = self.uh0.vec.FV().NumPy() > -40
        self.uh0.vec.FV().NumPy()[~self.pos] = -np.inf
        self.rho0.vec.FV().NumPy()[:] = np.exp(self.uh0.vec.FV().NumPy())

        if self.IC == 'waitingTime' or self.IC == 'BB':
            self.initialxRNode = np.where(self.uh0.vec.FV().NumPy()!=-np.inf)[0][-1]+1
            self.initialxLNode = np.where(self.uh0.vec.FV().NumPy()!=-np.inf)[0][0]-1
        

    
    def getPointsRhoex(self): return [self.rhoex(self.mesh(x)) for x in self.pnts_x]
    
    def createPlot(self):
        if self.plot:
            if self.dim == 1:
                self.fig = plt.figure()
                self.fig.set_size_inches(6, 4, forward=True)
                self.ax = self.fig.add_subplot(111)
                self.ax.set_ylabel(r'$\rho$',fontsize=12)
                self.ax.set_xlabel(r'$x$',fontsize=12)
                self.ax.minorticks_on()
                self.ax.tick_params(direction="in",which='both',axis='both',
                               bottom=True, top=True, left=True, right=True)
                self.pnts_rho = np.exp(self.uh0.vec)
                self.line1 = self.ax.plot(self.pnts_x,self.pnts_rho,'rs',label=r'$\rho$')[0]
                self.line2 = self.ax.plot(self.pnts_x, self.getPointsRhoex(),'k-',label=r'$\rho_{exact}$')[0]
                if self.plotInterface:  
                    self.line3 = self.ax.axvline(x=self.pnts_x[self.initialxLNode],linestyle='--',color='blue')
                    self.line4 = self.ax.axvline(x=self.pnts_x[self.initialxRNode],linestyle='--',color='blue')
                
                self.ax.legend(frameon=0,loc='upper left',fontsize=10)
                if self.IC == "accuracy": self.ax.set_ylim([1,1.6])
                self.fig.canvas.draw()
            else:
                self.scene = Draw(self.rho0,self.mesh)
    
    def reDrawPlot(self):
        if self.plot and self.step % self.plotEvery == 0:
            if self.dim == 1:
                self.pnts_rho_ex = [self.rhoex(self.mesh(x)) for x in self.pnts_x]
                self.line1.set_ydata(self.rhok.vec)
                self.line2.set_ydata(self.pnts_rho_ex)
                if self.plotInterface:
                    self.line3.set_xdata(self.getLeftInterface())
                    self.line4.set_xdata(self.getRightInterface())
                self.fig.canvas.draw()
            else:
                self.scene.Redraw()
    
    def takeTimestep(self):
        self.step += 1
        self.tstart.Set(self.tstart.Get()+self.dt.Get())
        self.Newton()
        self.uh0.vec.data = self.uh.vec
        self.rho0.vec.FV().NumPy()[:] = np.exp(self.uh0.vec.FV().NumPy())
    
    def simulate(self,tstop):
        with TaskManager():
            while self.tstart.Get() < tstop-self.dt.Get()/2:
                self.takeTimestep()
                self.reDrawPlot()
                if self.printTime:
                    print('\rTime=%.4f'%(self.tstart.Get()),end="")
                
    def Newton(self):
        self.uh.vec.data = self.uh0.vec
        self.rhok.vec.FV().NumPy()[:] = np.exp(self.uh.vec.FV().NumPy()[:])
        # set -inf to zero for simplicity!!! FIXME LATER
        self.pos = self.rhok.vec.FV().NumPy()==0
        self.uh.vec.FV().NumPy()[self.pos] = 0 
        self.rhouk.vec.FV().NumPy()[:] = self.rhok.vec.FV().NumPy()[:]*self.uh.vec.FV().NumPy()[:]

        count = 0
        ener0 = (self.uh.vec.FV().NumPy()[:]-1).dot(self.rhok.vec.FV().NumPy()[:])

        tol = abs(ener0)*self.newtonTolerance # relative tolerance
        while True:
            count += 1
            # hack zero
            self.a.Assemble()
            self.f.Assemble()
            if count==1: # locate active Dofs
                rows,cols,vals = self.a.mat.COO()
                A = sp.csr_matrix((vals,(rows,cols)))
                active = A.diagonal()>self.cutOff

                self.activeDofs = BitArray(active)
                if self.dcFlag == "left|right": 
                    self.activeDofs[0] = 0
                    self.activeDofs[-1] = 0


            self.uh.vec.data = self.a.mat.Inverse(freedofs = self.activeDofs,inverse="sparsecholesky")*self.f.vec
            self.rhok.vec.FV().NumPy()[active] = np.exp(self.uh.vec.FV().NumPy()[active])
            self.rhouk.vec.FV().NumPy()[active] = self.rhok.vec.FV().NumPy()[active]*self.uh.vec.FV().NumPy()[active]


            ener1 = (self.uh.vec.FV().NumPy()[active]-1).dot(self.rhok.vec.FV().NumPy()[active])
            err = abs(ener1-ener0)
            ener0 = ener1
            if err < tol:
                self.uh.vec.FV().NumPy()[~active] = -np.inf
                self.newtonCount.append(count)
                break
            if np.isnan(err) or count==30:
                print(count,"FAILED")
                stop 
                
    def getAvgNewtonCount(self): return sum(self.newtonCount)*1.0/self.step
    
    def getL2error(self,a):
        if self.dim == 1: sub0 = IfPos((x-a)*(x+a), 0, 1)
        if self.dim == 2: 
            r0 = (x**2+y**2)**0.5
            sub0 = IfPos(r0-a,0,1)
        if self.dim == 3:
            r0 = (x**2+y**2+z**2)**0.5
            sub0 = IfPos(r0-a,0,1)
        return sqrt(Integrate(sub0*(self.rhok-self.rhoex)**2, self.mesh))
    
    def getPointerror(self,p): return abs(self.rhok(self.mesh(p))-self.rhoex(self.mesh(p)))
    
    def generateVTU(self):
        error = sqrt((self.rhok-self.rhoex)**2)
        filename = "LD"+"_dim_"+str(self.dim)+"_"+self.IC+"_m_"+str(self.m)+"_"
        if self.quads: filename += "quads"
        else: filename += "trigs"
        vtk = VTKOutput(ma=self.mesh,
                coefs=[self.rhok,error],
                names = ["density",'error'],
                filename=filename,
                subdivision=5)
        vtk.Do()
    
    def getLeftInterface(self):
        loc = np.where(self.uh.vec.FV().NumPy()!=-np.inf)[0][0]
        return self.pnts_x[loc]
    def getRightInterface(self):
        loc = np.where(self.uh.vec.FV().NumPy()!=-np.inf)[0][-1]
        return self.pnts_x[loc]
   
    
    
