from ngsolve import *
from netgen.csg import *
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from netgen.geom2d import SplineGeometry, CSG2d,Rectangle
from ngsolve.meshes import MakeStructured2DMesh, Make1DMesh, MakeQuadMesh, MakeStructured3DMesh
from math import isnan
import time
from ngsolve.nonlinearsolvers import Newton
from ngsolve.webgui import Draw
ngsglobals.msg_level = 0

class Mixed:
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
        if self.dim !=1: self.DGAverageHack()
    
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
            ir = IntegrationRule(points = [(0,0), (1,0)], weights = [1/2, 1/2] )
            self.ELE = SEGM
            self.dx = dx(intrules={self.ELE:ir})
        if self.dim ==2:
            if self.quads:
                self.ELE = QUAD
                irx =  IntegrationRule(points = [(0,0.5), (1,0.5)], weights = [1/2, 1/2])
                iry = IntegrationRule(points = [(0.5,0), (0.5,1)], weights = [1/2, 1/2] )
                self.dx = dx(intrules={self.ELE:irx})
                self.dy = dx(intrules={self.ELE:iry})
        if self.dim == 3:
            raise Exception("Mixed method not implemented for 3D")
            
    def createFESpace(self):
        self.V = L2(self.mesh)
        self.W = HDiv(self.mesh) if self.dim != 1 else H1(self.mesh)
        # Used in upwinding
        self.H = FacetFESpace(self.mesh) if self.dim != 1 else H1(self.mesh)
        
        self.fes = FESpace([self.V, self.W]) 

        
    def TnT(self):
        (self.rho, self.u), (self.eta, self.v) = self.fes.TnT()
        self.p = self.m/(self.m-1)*self.rho**(self.m-1)
        
    def createGF(self):
        self.gfu = GridFunction(self.fes)
        self.rhoh, self.uh = self.gfu.components
        # Used to store data of previous iteration
        self.rho0 = GridFunction(self.V)
        self.u0  = GridFunction(self.W)
        self.rhoAvg = GridFunction(self.H)
        # Used for storing inverted mass matrix values
        self.MIfun = GridFunction(self.H)

        
    def createForms(self):
        self.a = BilinearForm(self.fes)
        self.n = specialcf.normal(self.mesh.dim)
        
        self.flux = self.u*self.n*IfPos(self.uh*self.n, self.rho0, 2*self.rhoAvg-self.rho0)
        
        self.a += (self.rho-self.rho0)/self.dt*self.eta*dx
        self.a += self.flux*self.eta*dx(element_boundary=True)
        
        if self.dim ==1:
            self.a += self.u*self.v*self.dx
        if self.dim ==2:
            if self.quads:
                self.a += self.u[0]*self.v[0]*self.dx
                self.a += self.u[1]*self.v[1]*self.dy
            else:
                invB = self.invertedMassMatrix()
                self.MIfun.vec[:] = invB[:]
                self.a += self.MIfun*self.u*self.n*self.v*self.n*dx(element_boundary=True)
        if self.dim == 1:
            self.a += -self.p*grad(self.v)*dx 
        else:
            self.a += -self.p*div(self.v)*dx 

    def DGAverageHack(self):
        # Needed for newton
        phi, psi = self.H.TnT()
        aDG = BilinearForm(self.H)
        aDG += phi*psi*dx(element_boundary=True)
        aDG.Assemble()
        self.invaDG = aDG.mat.CreateSmoother(self.H.FreeDofs())
        self.fDG = LinearForm(self.H)
        self.fDG += self.rhoh*psi*dx(element_boundary=True)
        
        
    def setInitialData(self):
        if  self.dim == 1: 
            r2 = x**2
            self.pnts_x  = np.linspace(self.xmin,self.xmax,self.nx+1) 
            h0 = (self.xmax-self.xmin)/self.nx 
            self.pnts_xC = np.linspace(self.xmin+0.5*h0,self.xmax-0.5*h0,self.nx)
        elif self.dim == 2: r2 = x**2+y**2
        else: raise Exception("Not available.")
        
        if self.IC == 'BB':
            val = self.s0-self.k*(self.m-1)/(2*self.dim*self.m)*r2/(self.tstart+1)**(2*self.k/self.dim)
            self.rhoex = (self.tstart+1)**(-self.k)*(IfPos(val, val**(1/(self.m-1)), 0))
        
        elif self.IC=='waitingTime' and self.dim==1:
            self.rhoex = IfPos((x-pi/2)*(x+pi/2), 0, ((self.m-1)/self.m*cos(x)**2)**(1/(self.m-1)) )
            self.tWait = 1/2/(self.m+1)
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
        
        self.rhoh.Set(self.rhoex)
        if self.dim == 1:
            p0 = self.m/(self.m-1)*self.rhoh.vec.FV().NumPy()**(self.m-1)
            self.uh.vec.FV().NumPy()[1:-1] = (p0[:-1]-p0[1:])/h0
            
        self.initialxRNode = np.where(self.rhoh.vec.FV().NumPy()>=1e-7)[0][-1]+1
        self.initialxLNode = np.where(self.rhoh.vec.FV().NumPy()>=1e-7)[0][0]-1
        
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

                self.line1 = self.ax.plot(self.pnts_xC,self.rhoh.vec,'rs',label=r'$\rho$')[0]
                self.line2 = self.ax.plot(self.pnts_x, self.getPointsRhoex(),'k-',label=r'$\rho_{exact}$')[0]
                if self.plotInterface:
                    
                    self.line3 = self.ax.axvline(x=self.pnts_xC[self.initialxLNode],linestyle='--',color='blue')
                    self.line4 = self.ax.axvline(x=self.pnts_xC[self.initialxRNode],linestyle='--',color='blue')
                
                self.ax.legend(frameon=0,loc='upper left',fontsize=10)
                if self.IC == "accuracy": self.ax.set_ylim([1,1.6])
                self.fig.canvas.draw()
            else:
                self.scene = Draw(self.rhoh,self.mesh)
    
    def reDrawPlot(self):
        if self.plot and self.step % self.plotEvery == 0:
            if self.dim == 1:
                self.pnts_rho_ex = [self.rhoex(self.mesh(x)) for x in self.pnts_x]
                self.line1.set_ydata(self.rhoh.vec)
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
        self.rho0.vec.data = self.rhoh.vec
        #self.u0.vec.data = self.uh.vec
        if self.dim == 1: self.rhoAvg.Set(self.rhoh) 
        else: 
            self.fDG.Assemble()
            self.rhoAvg.vec.data = self.invaDG*self.fDG.vec
        it = Newton(self.a, self.gfu, printing=False, maxit=30, maxerr=1e-12)
        self.newtonCount.append(it)
        self.reDrawPlot()
    
    def simulate(self,tstop):
        with TaskManager():
            while self.tstart.Get() < tstop-self.dt.Get()/2:
                self.takeTimestep()
                self.reDrawPlot()
                if self.printTime:
                    print('\rTime=%.4f'%(self.tstart.Get()),end="")
                
    
    def getL2error(self,a):
        if self.dim == 1: sub0 = IfPos((x-a)*(x+a), 0, 1)
        if self.dim == 2: 
            r0 = (x**2+y**2)**0.5
            sub0 = IfPos(r0-a,0,1)
        if self.dim == 3:
            r0 = (x**2+y**2+z**2)**0.5
            sub0 = IfPos(r0-a,0,1)
        return sqrt(Integrate(sub0*(self.rhoh-self.rhoex)**2, self.mesh))
    
    
    def generateVTU(self):
        error = sqrt((self.rhoh-self.rhoex)**2)
        filename = "Mixed"+"_dim_"+str(self.dim)+"_"+self.IC+"_m_"+str(self.m)+"_"
        if self.quads: filename += "quads"
        else: filename += "trigs"
        vtk = VTKOutput(ma=self.mesh,
                coefs=[self.rhoh,error],
                names = ["density",'error'],
                filename=filename,
                subdivision=5)
        vtk.Do()
    
    def getLeftInterface(self):
        loc = np.where(self.rhoh.vec.FV().NumPy()>=1e-7)[0][0]
        return self.pnts_xC[loc]
    def getRightInterface(self):
        loc = np.where(self.rhoh.vec.FV().NumPy()>=1e-7)[0][-1]
       
        return self.pnts_xC[loc]
    
    def invertedMassMatrix(self):
        mesh = self.mesh
        vertices={}
        for v in mesh.vertices:
            vertices[v] = v.point
        edges = {}

        for e in mesh.edges:
            edges[e] = e.vertices

        angles = {}
        lengths = {}
        for el in mesh.Elements():

            lengloc = {}
            for e in el.edges:
                    x1 = vertices[edges[e][0]][0]
                    y1 = vertices[edges[e][0]][1]
                    x2 = vertices[edges[e][1]][0]
                    y2 = vertices[edges[e][1]][1]

                    dx = (x1-x2)
                    dy = (y1-y2)
                    l = (np.sqrt((dx)**2+(dy)**2))
                    lengths[e] = l
                    lengloc[e] = l


            for e in el.edges:
                elist = list(lengloc.keys())
                elist.remove(e)
                c = lengths[e]
                a = lengths[elist[0]]
                b = lengths[elist[1]]
                theta = np.arccos((a**2 + b**2 -c**2)/2/a/b)
                try:
                    angles[e].append(theta)
                except:
                    angles[e] = [theta]


        sortedAngles = dict(sorted(angles.items(),key=lambda t: t[0].nr ))
        invB = []
        B = []
        for key,val in sortedAngles.items():
            l = lengths[key]
            entry = (0.5*np.sum(1/np.tan(sortedAngles[key])))
            invB.append(0.5*l*1/entry)
            B.append(0.5*l*entry)

        return np.array(B)
    