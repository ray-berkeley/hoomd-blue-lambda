// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

/*!
 * \file mpcd/Communicator.h
 * \brief Defines the mpcd::Communicator class
 */

#ifdef ENABLE_MPI

#ifndef MPCD_COMMUNICATOR_H_
#define MPCD_COMMUNICATOR_H_

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include "CommunicatorUtilities.h"
#include "ParticleData.h"
#include "SystemData.h"

#include "hoomd/HOOMDMath.h"
#include "hoomd/GPUArray.h"
#include "hoomd/GPUVector.h"
#include "hoomd/DomainDecomposition.h"
#include "hoomd/Autotuner.h"

#include <memory>
#include <vector>
#include "hoomd/extern/nano-signal-slot/nano_signal_slot.hpp"
#include "hoomd/extern/pybind/include/pybind11/pybind11.h"

//! Forward declarations for some classes
class SystemDefinition;
class Profiler;
struct BoxDim;
class ParticleData;

namespace mpcd
{

//! MPI communication of MPCD particle data
/*!
 * This class implements the communication algorithms for mpcd::ParticleData that
 * are used in parallel simulations on the CPU. A domain decomposition communication pattern
 * is used so that every processor owns particles that are spatially local (\cite Plimpton 1995). So
 * far, the only communication needed for MPCD particles is migration, which is handled
 * using the same algorithms as for the standard ::ParticleData (::Communicator::migrateParticles).
 *
 * There is unfortunately significant code duplication with ::Communicator, but
 * there is little that can be done about this without creating an abstracted
 * communication base class.
 *
 * \ingroup communication
 */
class Communicator
    {
    public:
        //! Constructor
        Communicator(std::shared_ptr<mpcd::SystemData> system_data);

        //! Destructor
        virtual ~Communicator();

        //! \name accessor methods
        //@{

        //! Get the number of unique neighbors
        unsigned int getNUniqueNeighbors() const
            {
            return m_n_unique_neigh;
            }

        //! Get the array of unique neighbors
        const GPUArray<unsigned int>& getUniqueNeighbors() const
            {
            return m_unique_neighbors;
            }

        //! Set autotuner parameters
        /*!
         * \param enable Enable/disable autotuning
         * \param period period (approximate) in time steps when returning occurs
         */
        virtual void setAutotunerParams(bool enable, unsigned int period)
            { }

        //! Set the profiler.
        /*!
         * \param prof Profiler to use with this class
         */
        void setProfiler(std::shared_ptr<Profiler> prof)
            {
            m_prof = prof;
            }

        //@}

        //! \name communication methods
        //@{

        //! Interface to the communication methods
        /*!
         * This method is supposed to be called every time step and automatically performs all necessary
         * communication steps.
         */
        void communicate(unsigned int timestep);

        //! Migrate particle data to local domain
        /*!
         * This methods finds all the particles that are no longer inside the domain
         * boundaries and transfers them to neighboring processors.
         *
         * Particles sent to a neighbor are deleted from the local particle data.
         * Particles received from a neighbor in one of the six communication steps
         * are added to the local particle data, and are also considered for forwarding to a neighbor
         * in the subseqent communication steps.
         *
         * \post Every particle on every processor can be found inside the local domain boundaries.
         */
        virtual void migrateParticles();

        //@}

    protected:
        //! Set the communication flags for the particle data
        virtual void setCommFlags(const BoxDim& box);

        //! Checks for overdecomposition
        void checkDecomposition();

        //! Get the wrapping box for this rank
        BoxDim getWrapBox(const BoxDim& box);

        //! Returns true if we are communicating particles along a given direction
        /*!
         * \param dir Direction to return dimensions for
         */
        bool isCommunicating(mpcd::detail::face dir) const
            {
            const Index3D& di = m_decomposition->getDomainIndexer();
            bool res = true;
            if ((dir == mpcd::detail::face::east || dir == mpcd::detail::face::west) && di.getW() == 1)
                res = false;
            if ((dir == mpcd::detail::face::north || dir == mpcd::detail::face::south) && di.getH() == 1)
                res = false;
            if ((dir == mpcd::detail::face::up || dir == mpcd::detail::face::down) && di.getD() == 1)
                res = false;

            return res;
            }

        std::shared_ptr<mpcd::SystemData> m_mpcd_sys;               //!< MPCD system data
        std::shared_ptr<SystemDefinition> m_sysdef;                 //!< HOOMD system definition
        std::shared_ptr<::ParticleData> m_pdata;                    //!< HOOMD particle data
        std::shared_ptr<const ExecutionConfiguration> m_exec_conf;  //!< Execution configuration
        std::shared_ptr<mpcd::ParticleData> m_mpcd_pdata;           //!< MPCD particle data
        const MPI_Comm m_mpi_comm;                                  //!< MPI communciator
        std::shared_ptr<DomainDecomposition> m_decomposition;       //!< Domain decomposition information
        std::shared_ptr<Profiler> m_prof;                           //!< Profiler

        bool m_is_communicating;               //!< Whether we are currently communicating
        bool m_force_migrate;                  //!< True if particle migration is forced
        bool m_check_decomposition; //!< Flag to check the simulation box decomposition

        const static unsigned int neigh_max;           //!< Maximum number of neighbor ranks
        GPUArray<unsigned int> m_neighbors;            //!< Neighbor ranks
        GPUArray<unsigned int> m_unique_neighbors;     //!< Neighbor ranks w/duplicates removed
        GPUArray<unsigned int> m_adj_mask;             //!< Adjacency mask for every neighbor
        unsigned int m_nneigh;                         //!< Number of neighbors
        unsigned int m_n_unique_neigh;                 //!< Number of unique neighbors

        //! Helper function to initialize adjacency arrays
        void initializeNeighborArrays();

        GPUVector<mpcd::detail::pdata_element> m_sendbuf;   //!< Buffer for particles that are sent
        GPUVector<mpcd::detail::pdata_element> m_recvbuf;   //!< Buffer for particles that are received
        std::vector<MPI_Request> m_reqs;    //!< MPI requests
        std::vector<MPI_Status> m_stats;    //!< MPI statuses

    private:
        //! Notify communicator that box has changed and so decomposition needs to be checked
        void slotBoxChanged()
            {
            m_check_decomposition = true;
            }
    };


namespace detail
{
//! Export mpcd::Communicator to python
void export_Communicator(pybind11::module& m);
} // end namespace detail

} // end namespace mpcd

#endif // MPCD_COMMUNICATOR_H_
#endif // ENABLE_MPI
