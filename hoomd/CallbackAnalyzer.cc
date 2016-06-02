// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: csadorf,samnola

/*! \file CallbackAnalyzer.cc
    \brief Defines the CallbackAnalyzer class
*/



#include "CallbackAnalyzer.h"

#ifdef ENABLE_MPI
#include "Communicator.h"
#endif

#include <boost/python.hpp>
using namespace boost::python;

#include <iomanip>
using namespace std;

/*! \param sysdef SystemDefinition containing the Particle data to analyze
    \param callback A python functor object to be used as callback
*/
CallbackAnalyzer::CallbackAnalyzer(boost::shared_ptr<SystemDefinition> sysdef,
                         boost::python::object callback)
    : Analyzer(sysdef), callback(callback)
    {
    m_exec_conf->msg->notice(5) << "Constructing CallbackAnalyzer" << endl;
    }

CallbackAnalyzer::~CallbackAnalyzer()
    {
    m_exec_conf->msg->notice(5) << "Destroying CallbackAnalyzer" << endl;
    }

/*!\param timestep Current time step of the simulation

    analyze() will call the callback
*/
void CallbackAnalyzer::analyze(unsigned int timestep)
    {
      callback(timestep);
    }

void export_CallbackAnalyzer()
    {
    class_<CallbackAnalyzer, boost::shared_ptr<CallbackAnalyzer>, bases<Analyzer>, boost::noncopyable>
    ("CallbackAnalyzer", init< boost::shared_ptr<SystemDefinition>, boost::python::object>())
    ;
    }