# Test npt updater for basic sanity. Proper validation requires thermodynamic data from much longer runs.
from __future__ import division, print_function
from hoomd import *
from hoomd import hpmc
import unittest
import numpy as np

context.initialize()

def create_empty(**kwargs):
    snap = data.make_snapshot(**kwargs);
    return init.read_snapshot(snap);

class npt_sanity_checks (unittest.TestCase):
    # This test runs a system at high enough pressure and for enough steps to ensure a dense system.
    # After adequate compression, it confirms at 1000 different steps in the simulation, the NPT
    # updater does not introduce overlaps.
    def test_prevents_overlaps(self):
        N=64
        L=20
        self.system = create_empty(N=N, box=data.boxdim(L=L, dimensions=2), particle_types=['A'])
        self.mc = hpmc.integrate.convex_polygon(seed=1, d=0.1, a=0.1)
        self.npt = hpmc.update.npt(self.mc, seed=1, P=1000, dLx=0.01, dLy=0.01, dLz=0.01, dxy=0.01, dxz=0.01, dyz=0.01)
        self.mc.shape_param.set('A', vertices=[(-1,-1), (1,-1), (1,1), (-1,1)])

        # place particles
        a = L / 8.
        for k in range(N):
            i = k % 8
            j = k // 8 % 8
            self.system.particles[k].position = (i*a - 9.9, j*a - 9.9, 0)

        run(0)
        self.assertEqual(self.mc.count_overlaps(), 0)
        run(1000)
        overlaps = 0
        for i in range(100):
            run(10, quiet=True)
            overlaps += self.mc.count_overlaps()
        self.assertEqual(overlaps, 0)
        print(self.system.box)

        del self.npt
        del self.mc
        del self.system
        context.initialize()

    # This test places two particles that overlap significantly.
    # The maximum move displacement is set so that the overlap cannot be removed.
    # It then performs an NPT run and ensures that no volume or shear moves were accepted.
    def test_rejects_overlaps(self):
        self.system = create_empty(N=2, box=data.boxdim(L=100), particle_types=['A'])
        self.mc = hpmc.integrate.convex_polyhedron(seed=1, d=0.1, a=0.1,max_verts=8)
        self.npt = hpmc.update.npt(self.mc, seed=1, P=1000, dLx=0.1, dLy=0.1, dLz=0.1, dxy=0.01, dxz=0.01, dyz=0.01)
        self.mc.shape_param.set('A', vertices=[  (1,1,1), (1,-1,1), (-1,-1,1), (-1,1,1),
                                            (1,1,-1), (1,-1,-1), (-1,-1,-1), (-1,1,-1) ])

        self.system.particles[1].position = (0.7,0,0)

        run(0)
        overlaps = self.mc.count_overlaps()
        self.assertGreater(overlaps, 0)

        run(100)
        self.assertEqual(overlaps, self.mc.count_overlaps())
        self.assertEqual(self.npt.get_volume_acceptance(), 0)
        self.assertEqual(self.npt.get_shear_acceptance(), 0)

        del self.npt
        del self.mc
        del self.system
        context.initialize() 
    
    # This test runs a single-particle NPT system to test whether NPT allows the box to invert.
#    def test_box_inversion(self):
#        for i in range(5):
#            self.system = create_empty(N=1, box=data.boxdim(L=4), particle_types=['A'])
#            self.mc = hpmc.integrate.sphere(seed=i, d=0.0)
#            self.npt = hpmc.update.npt(self.mc, seed=1, P=100, dLx=10.0, dLy=10.0, dLz=10.0, dxy=0, dxz=0, dyz=0, move_ratio=1)
#            self.mc.shape_param.set('A', diameter=1.0)
#
#            self.system.particles[0].position = (0,0,0)
#
#            for j in range(10):
#                run(10)
#                self.assertGreater(self.system.box.get_volume(), 0)
#                print(self.system.box)
#
#            del self.npt
#            del self.mc
#            del self.system
#            context.initialize()

# This test takes too long to run. Validation tests do not need to be run on every commit.
# class npt_thermodynamic_tests (unittest.TestCase):
#     # This test checks the NPT updater against the ideal gas equation of state
#     def test_ideal_gas(self):
#         N=100
#         L=2.0
#         nsteps = 1e4
#         nsamples = 1e3
#         sample_period = int(nsteps/nsamples)
#         class accumulator:
#             def __init__(self,nsamples,system):
#                 self.volumes = np.empty((nsamples),)
#                 self.i = 0
#                 self.system = system
#             def callback(self,timestep):
#                 if self.i < nsamples:
#                     self.volumes[self.i] = self.system.box.get_volume()
#                 self.i += 1
#             def get_volumes(self):
#                 return self.volumes[:self.i]
#         system = create_empty(N=N, box=data.boxdim(L=L, dimensions=3), particle_types=['A'])
#         mc = hpmc.integrate.sphere(seed=1, d=0.0)
#         npt = hpmc.update.npt(mc, seed=1, P=N, dLx=0.2, move_ratio=1.0, isotropic=True)
#         mc.shape_param.set('A', diameter=0.0)

#         # place particles
#         positions = np.random.random((N,3))*L - 0.5*L
#         for k in range(N):
#             system.particles[k].position = positions[k]

#         my_acc = accumulator(nsamples, system)
#         run(1e5, callback_period=sample_period, callback=my_acc.callback)
#         # for beta P == N the ideal gas law says V must be 1.0. We'll grant 10% error
#         self.assertLess(np.abs(my_acc.get_volumes().mean() - 1.0), 0.1)

#         del my_acc
#         del npt
#         del mc
#         del system
#         context.initialize()


class boxMC_sanity_checks (unittest.TestCase):
    # This test runs a system at high enough pressure and for enough steps to ensure a dense system.
    # After adequate compression, it confirms at 1000 different steps in the simulation, the NPT
    # updater does not introduce overlaps.
    def test_prevents_overlaps(self):
        N=64
        L=20
        self.snapshot = data.make_snapshot(N=N, box=data.boxdim(L=L, dimensions=2), particle_types=['A'])
        self.system = init.read_snapshot(self.snapshot)
        self.mc = hpmc.integrate.convex_polygon(seed=1, d=0.1, a=0.1)
        self.boxMC = hpmc.update.boxMC(self.mc, betaP=1000, seed=1)
        self.mc.shape_param.set('A', vertices=[(-1,-1), (1,-1), (1,1), (-1,1)])

        # place particles
        a = L / 8.
        for k in range(N):
            i = k % 8
            j = k // 8 % 8
            self.system.particles[k].position = (i*a - 9.9, j*a - 9.9, 0)

        run(0)
        self.assertEqual(self.mc.count_overlaps(), 0)
        run(1000)
        overlaps = 0
        for i in range(100):
            run(10, quiet=True)
            overlaps += self.mc.count_overlaps()
        self.assertEqual(overlaps, 0)
        #print(system.box)

        del self.boxMC
        del self.mc
        del self.system
        del self.snapshot
        context.initialize()

    # This test places two particles that overlap significantly.
    # The maximum move displacement is set so that the overlap cannot be removed.
    # It then performs an NPT run and ensures that no volume or shear moves were accepted.
    def test_rejects_overlaps(self):
        self.snapshot = data.make_snapshot(N=2, box=data.boxdim(L=4), particle_types=['A'])
        self.system = init.read_snapshot(self.snapshot)
        self.mc = hpmc.integrate.convex_polyhedron(seed=1, d=0.1, a=0.1)
        self.boxMC = hpmc.update.boxMC(self.mc, betaP=1000, seed=1)
        self.mc.shape_param.set('A', vertices=[  (1,1,1), (1,-1,1), (-1,-1,1), (-1,1,1),
                                            (1,1,-1), (1,-1,-1), (-1,-1,-1), (-1,1,-1) ])

        self.system.particles[1].position = (0.7,0,0)

        run(0)
        overlaps = self.mc.count_overlaps()
        self.assertGreater(overlaps, 0)

        run(100)
        self.assertEqual(overlaps, mc.count_overlaps())
        self.assertEqual(self.boxMC.get_volume_acceptance(), 0)
        self.assertEqual(self.boxMC.get_shear_acceptance(), 0)

        del self.boxMC
        del self.mc
        del self.system
        del self.snapshot
        context.initialize()
'''
    # This test runs a single-particle NPT system to test whether NPT allows the box to invert.
    def test_VolumeMove_box_inversion(self):
        for i in range(5):
            snapshot = data.make_snapshot(N=1, box=data.boxdim(L=0.1), particle_types=['A'])
            snapshot.particles.position[:] = (0,0,0)
            system = init.read_snapshot(snapshot)
            mc = hpmc.integrate.sphere(seed=i, d=0.0)
            boxMC = hpmc.update.boxMC(mc, betaP=100, seed=1)
            boxMC.setVolumeMove(delta=1.0)
            mc.shape_param.set('A', diameter=0.0)

            for j in range(10):
                run(10)
                self.assertGreater(system.box.get_volume(), 0)
                #print(system.box)

            del boxMC
            del mc
            del system
            del snapshot
            context.initialize()

    # This test runs a single-particle NPT system to test whether NPT allows the box to invert.
    def test_LengthMove_box_inversion(self):
        for i in range(5):
            snapshot = data.make_snapshot(N=1, box=data.boxdim(L=0.1), particle_types=['A'])
            snapshot.particles.position[:] = (0,0,0)
            system = init.read_snapshot(snapshot)
            mc = hpmc.integrate.sphere(seed=i, d=0.0)
            boxMC = hpmc.update.boxMC(mc, betaP=100, seed=1)
            boxMC.setLengthMove(delta=[1.0, 1.0, 1.0])
            mc.shape_param.set('A', diameter=0.0)

            for j in range(10):
                run(10)
                self.assertGreater(system.box.get_volume(), 0)
                #print(system.box)

            del boxMC
            del mc
            del system
            del snapshot
            context.initialize()

# These tests check the methods for functionality
class boxMC_test_methods (unittest.TestCase):
    def setUp(self):
        snapshot = data.make_snapshot(N=1, box=data.boxdim(L=4), particle_types=['A'])
        snapshot.particles.position[0] = (0, 0, 0)
        self.system = init.read_snapshot(snapshot)
        self.mc = hpmc.integrate.sphere(seed=1)
        self.mc.shape_param.set('A', diameter = 1.0)
        self.boxMC = hpmc.update.boxMC(self.mc, betaP=100, seed=1)

    def tearDown(self):
        del self.boxMC
        del self.mc
        del self.system
        context.initialize();

    def test_methods_setVolumeMove(self):
        boxMC = self.boxMC
        boxMC.setVolumeMove(delta=1.0)
        boxMC.setVolumeMove(delta=1.0, weight=1)

    def test_warnings_setVolumeMove(self):
        boxMC = self.boxMC
        success = True
        # Catch all warnings/errors associated with setVolumeMove(self, delta=None, weight=1.0)
        try:boxMC.setVolumeMove(delta=None)
        except ValueError: print('Raised correct error: Volume move undefined.')
        else:
            print('No error detected for VolumeMove delta=None')
            success = False
        try: boxMC.setVolumeMove(delta = 10.0, weight=None)
        except ValueError: print('Raised correct error: Volume weight undefined.')
        else:
            print('No error detected for VolumeMove weight=None')
            success = False
        self.assertEqual(success, True)

    def test_methods_setLengthMove(self):
        boxMC = self.boxMC
        # test scalar delta
        boxMC.setLengthMove(delta=10.0)
        # test list delta
        boxMC.setLengthMove(delta=(1,1,1))
        boxMC.setLengthMove(delta=(1,1,1), weight=2)

    def test_warnings_setLengthMove(self):
        boxMC = self.boxMC
        success = True
        # Catch all warnings/errors associated with setLengthMove(self, delta=None, weight=1.0)
        try:boxMC.setLengthMove(delta=None)
        except ValueError: print('Raised correct error: Length move undefined.')
        else:
            print('No error detected for LengthMove delta=None')
            success = False
        try: boxMC.setLengthMove(delta=10.0, weight=None)
        except ValueError: print('Raised correct error: Length weight undefined.')
        else:
            print('No error detected for LengthMove delta=None')
            success = False
        self.assertEqual(success, True)

    def test_methods_setShearMove(self):
        boxMC = self.boxMC
        # test scalar delta
        boxMC.setShearMove(delta=1.0)
        # test list delta
        boxMC.setShearMove(delta=(1,1,1))
        boxMC.setShearMove(delta=(1,1,1), weight=2)

    def test_methods_setShearMove(self):
        boxMC = self.boxMC
        success = True
        # Catch all warnings/errors associated with setLengthMove(self, delta=None, weight=1.0)
        try:boxMC.setShearMove(delta=None)
        except ValueError: print('Raised correct error: Shear move undefined.')
        else:
            print('No error detected for ShearMove delta=None')
            success = False
        try: boxMC.setShearMove(delta=10.0, weight=None)
        except ValueError: print('Raised correct error: Shear weight undefined.')
        else:
            print('No error detected for ShearMove weight=None')
            success = False
        try: boxMC.setShearMove(delta=10.0, reduce=None)
        except ValueError: print('Raised correct error: Shear reduction undefined.')
        else:
            print('No error detected for ShearMove reduce=None')
            success = False
        import warnings
        self.assertEqual(success, True)

    def test_methods_get_misc(self):
        boxMC = self.boxMC
        self.assertNotEqual(boxMC.get_params(), False)
        self.assertGreater(boxMC.get_betaP(), 0)
        #self.assertNotEqual(boxMC.get_delta(), None)
        #self.assertNotEqual(boxMC.get_volume, None)

# This test takes too long to run. Validation tests do not need to be run on every commit.
# class boxMC_thermodynamic_tests (unittest.TestCase):
#     # This test checks the BoxMC updater against the ideal gas equation of state
#     def test_volume_ideal_gas(self):
#         N=100
#         L=2.0
#         nsteps = 1e4
#         nsamples = 1e3
#         sample_period = int(nsteps/nsamples)
#         class accumulator:
#             def __init__(self,nsamples,system):
#                 self.volumes = np.empty((nsamples),)
#                 self.i = 0
#                 self.system = system
#             def callback(self,timestep):
#                 if self.i < nsamples:
#                     self.volumes[self.i] = self.system.box.get_volume()
#                 self.i += 1
#             def get_volumes(self):
#                 return self.volumes[:self.i]
#         self.system = create_empty(N=N, box=data.boxdim(L=L, dimensions=3), particle_types=['A'])
#         self.mc = hpmc.integrate.sphere(seed=1, d=0.0)
#         npt = hpmc.update.boxMC(self.mc, betaP=N, seed=1)
#         npt.setVolumeMove(delta=0.2, weight=1)
#         self.mc.shape_param.set('A', diameter=0.0)

#         # place particles
#         positions = np.random.random((N,3))*L - 0.5*L
#         for k in range(N):
#             self.system.particles[k].position = positions[k]

#         my_acc = accumulator(nsamples, self.system)
#         run(1e5, callback_period=sample_period, callback=my_acc.callback)
#         # for beta P == N the ideal gas law says V must be 1.0. We'll grant 10% error
#         self.assertLess(np.abs(my_acc.get_volumes().mean() - 1.0), 0.1)

#         del my_acc
#         del npt
#         del self.mc
#         del self.system
#         context.initialize()
'''
if __name__ == '__main__':
    # this test works on the CPU only and only on a single rank
    if comm.get_num_ranks() > 1:
        raise RuntimeError("This test only works on 1 rank");

    unittest.main(argv = ['test.py', '-v'])
