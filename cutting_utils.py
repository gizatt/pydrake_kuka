import numpy as np
import time

from pydrake.all import (
    AbstractValue,
    LeafSystem,
    PortDataType,
    RigidBodyPlant,
    RigidBodyTree
)

class CutException(Exception):
    def __init__(self, cut_body_index, cut_pt, cut_normal):
        # Call the base class constructor with the parameters it needs
        super(CutException, self).__init__("Cut was triggered.")
        self.cut_body_index = cut_body_index
        # Both of these should be in *body frame* of the cut object.
        self.cut_pt = cut_pt
        self.cut_normal = cut_normal


class CuttingGuard(LeafSystem):
    def __init__(self, rbt, rbp, cutting_body_index, cut_direction,
                 cut_normal, min_cut_force, cuttable_body_indices,
                 timestep=0.01, name="Cutting Guard",
                 last_cut_time = 0.):
        ''' Watches the RBT contact results output, and
        raises an exception (to pause simulation). '''
        LeafSystem.__init__(self)
        self.set_name(name)

        self._DeclarePeriodicPublish(timestep, 0.0)
        self.rbt = rbt
        self.rbp = rbp
        self.last_cut_time = last_cut_time

        self.collision_id_to_body_index_map = {}
        for k in range(self.rbt.get_num_bodies()):
            for i in rbt.get_body(k).get_collision_element_ids():
                self.collision_id_to_body_index_map[i] = k

        self.cutting_body_index = cutting_body_index
        self.cutting_body_ids = rbt.get_body(cutting_body_index).get_collision_element_ids()
        self.cut_direction = np.array(cut_direction)
        self.cut_direction /= np.linalg.norm(self.cut_direction)
        self.cut_normal = np.array(cut_normal)
        self.cut_normal /= np.linalg.norm(self.cut_normal)
        self.min_cut_force = min_cut_force
        self.cuttable_body_indices = cuttable_body_indices

        self.state_input_port = \
            self._DeclareInputPort(PortDataType.kVectorValued,
                                   rbt.get_num_positions() +
                                   rbt.get_num_velocities())
        self.contact_results_input_port = \
            self._DeclareInputPort(PortDataType.kAbstractValued,
                                   rbp.contact_results_output_port().size())

    def _DoPublish(self, context, events):
        contact_results = self.EvalAbstractInput(
            context, self.contact_results_input_port.get_index()).get_value()
        x = self.EvalVectorInput(
                context, self.state_input_port.get_index()).get_value()

        if context.get_time() - self.last_cut_time < 0.01:
            return

        #print context.get_time(), x
        #time.sleep(0.25)
        # Check each contact to see if it's with the blade.
        this_contact_info = []
        for contact_i in range(contact_results.get_num_contacts()):
            contact_info = contact_results.get_contact_info(contact_i)
            contact_force = contact_info.get_resultant_force()
            cut_body_index = None
            cut_pt = contact_force.get_application_point()
            if contact_info.get_element_id_1() in self.cutting_body_ids:
                cut_body_index = self.collision_id_to_body_index_map[
                    contact_info.get_element_id_2()]
                cut_force = contact_force.get_force()
            elif contact_info.get_element_id_2() in self.cutting_body_ids:
                cut_body_index = self.collision_id_to_body_index_map[
                    contact_info.get_element_id_1()]
                cut_force = -contact_force.get_force()

            # If we contacted the blade and it's with a cuttable object,
            # check if we cut hard enough.
            if cut_body_index in self.cuttable_body_indices:
                # Point and force are in *world* frame
                # So see them in knife frame
                kinsol = self.rbt.doKinematics(x[0:self.rbt.get_num_positions()])
                tf = self.rbt.relativeTransform(kinsol, self.cutting_body_index, 0)
                knife_body_force = tf[0:3, 0:3].dot(cut_force)
                knife_body_pt = tf[0:3, 0:3].dot(cut_pt) + tf[0:3, 3]
                print "Got potential cut with body %d: " % cut_body_index, " force ", knife_body_force
                if -knife_body_force.dot(self.cut_direction) > self.min_cut_force:
                    # Trigger a cut exception!
                    tf = self.rbt.relativeTransform(kinsol, cut_body_index, self.cutting_body_index)
                    cut_body_pt = tf[0:3, 0:3].dot(knife_body_pt) + tf[0:3, 3]
                    cut_body_normal = tf[0:3, 0:3].dot(self.cut_normal)
                    print tf, self.cut_normal, cut_body_normal

                    # Further TF
                    print "Triggering cut!"
                    raise CutException(cut_body_index=cut_body_index,
                                       cut_pt=cut_body_pt, cut_normal=cut_body_normal)


if __name__ == "__main__":
    print "Goodbye"