// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

/*! \file Enforce2DUpdater.h
    \brief Declares an updater that zeros the momentum of the system
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include "hoomd/Updater.h"

#include <boost/shared_ptr.hpp>
#include <vector>

#ifndef __ENFORCE2DUPDATER_H__
#define __ENFORCE2DUPDATER_H__

//! Confines particles to the xy plane
/*! This updater zeros the z-velocities and z-forces to constrain particles
    to the xy plane.
    \ingroup updaters
*/
class Enforce2DUpdater : public Updater
    {
    public:
        //! Constructor
        Enforce2DUpdater(boost::shared_ptr<SystemDefinition> sysdef);

        //! Destructor
        virtual ~Enforce2DUpdater();

        //! Take one timestep forward
        virtual void update(unsigned int timestep);
    };

//! Export the Enforce2DUpdater to python
void export_Enforce2DUpdater();

#endif