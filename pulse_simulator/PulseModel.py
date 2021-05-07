import numpy as np
import pyro
import pyro.distributions as dist
import torch
from torch import nn
from torch.distributions import constraints
from pulse_simulator.utils.Networks import Net
import pyro.contrib.examples.polyphonic_data_loader as poly
from pulse_simulator.utils.MIMICDataset import (
    s0_cols,
    dummy_cols as cols,
    static_cols,
    action_cols,
    request_dict,
    get_data_req_mgr,
    column_mappings,
)
from simple_model.Policy import Policy
from max_likelihood.utils.HelperNetworks import Combiner
from pulse.cpm.PulsePhysiologyEnginePool import PulsePhysiologyEnginePool
from pulse.cpm.PulsePhysiologyEngine import PulsePhysiologyEngine
from pulse.cdm.engine import SEDataRequestManager, SEDataRequest, SEAdvanceTime
from pulse.cdm.patient import SEPatientConfiguration, SEPatient, eSex
from pulse.cdm.patient_actions import (
    SEHemorrhage,
    eHemorrhageType,
    SESubstanceBolus,
    eSubstance_Administration,
    SESubstanceCompoundInfusion,
    SESubstanceInfusion,
)
from pulse.cdm.scalars import (
    MassPerVolumeUnit,
    TimeUnit,
    VolumeUnit,
    MassUnit,
    PressureUnit,
    FrequencyUnit,
    VolumePerTimeUnit,
    LengthUnit,
)
from pulse_simulator.utils.PatientUtils import reset_extreme_readings
import logging

class PulseModel(nn.Module):
    def __init__(
        self, rnn_dim=40, rnn_dropout_rate=0.0, st_vec_dim=11, n_act=2, s0_vec_dim=len(s0_cols), use_cuda=False
    ):
        super().__init__()
        self.use_cuda = use_cuda
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.st_vec_dim = st_vec_dim
        self.s0_gender = nn.Parameter(torch.zeros(2))
        self.s0_age = nn.Parameter(torch.zeros(2, 2))
        self.s0_weight = Net(input_dim=2, output_dim=1)
        self.s0_Network = Net(input_dim=len(static_cols), output_dim=s0_vec_dim)
        self.vaso = Policy(
            input_dim=st_vec_dim + len(static_cols),
            hidden_1_dim=20,
            hidden_2_dim=20,
            output_dim=1,
            use_cuda=use_cuda,
        )
        self.iv = Policy(
            input_dim=st_vec_dim + len(static_cols),
            hidden_1_dim=20,
            hidden_2_dim=20,
            output_dim=1,
            use_cuda=use_cuda,
        )
        self.mechvent = Policy(
            input_dim=st_vec_dim + len(static_cols),
            hidden_1_dim=20,
            hidden_2_dim=20,
            output_dim=2,
            use_cuda=use_cuda,
        )
        self.rnn = nn.RNN(
            input_size=len(cols),
            hidden_size=rnn_dim,
            nonlinearity="tanh",
            batch_first=True,
            bidirectional=False,
            num_layers=1,
            dropout=rnn_dropout_rate,
        )
        self.h_0 = nn.Parameter(torch.zeros(1, 1, rnn_dim))
        self.combiner = Combiner(
            z_dim=len(static_cols), rnn_dim=rnn_dim, out_dim=s0_vec_dim
        )
        self.s_q_0 = nn.Parameter(torch.zeros(st_vec_dim))
        if use_cuda:
            self.cuda()

    def emission(self, pool, t, mini_batch, mini_batch_mask):
        data, active = self.pull_data(pool, len(mini_batch))
        requests = list(request_dict.keys())
        for column, request in column_mappings.items():
            pyro.sample(
                f"x{t}_{column}",
                dist.Normal(data[:, requests.index(request)], 0.001).mask(mini_batch_mask[:, t] * active),
                obs=mini_batch[:, t, cols.index(column)],
            )
        return data, active

    def create_pool(self, st_gender, st_age, st_weight, s0):
        batch_size = st_gender.size(0)
        pool = PulsePhysiologyEnginePool(
            0, "/data/localhost/taufiq/Pulse/builds/install/bin"
        )
        for i in range(batch_size):
            pe1 = pool.create_engine(i + 1)
            pe1.engine_initialization.data_request_mgr = get_data_req_mgr()
            pe1.engine_initialization.patient_configuration = SEPatientConfiguration()
            patient = pe1.engine_initialization.patient_configuration.get_patient()
            patient.set_name(str(i))
            if st_gender[i].detach().item() == 1:
                patient.set_sex(eSex.Female)
            patient.get_weight().set_value(st_weight[i].detach().item(), MassUnit.kg)
            patient.get_age().set_value(st_age[i].detach().item(), TimeUnit.day)
            patient.get_heart_rate_baseline().set_value(
                s0[i, s0_cols.index("HR")].detach().item(), FrequencyUnit.Per_min
            )
            patient.get_systolic_arterial_pressure_baseline().set_value(s0[i, s0_cols.index("SysBP")].detach().item(), PressureUnit.mmHg)
            patient.get_diastolic_arterial_pressure_baseline().set_value(s0[i, s0_cols.index("DiaBP")].detach().item(), PressureUnit.mmHg)
            patient.get_respiration_rate_baseline().set_value(s0[i, s0_cols.index("RR")].detach().item(), FrequencyUnit.Per_min)
            patient.get_height().set_value(s0[i, s0_cols.index("height")].detach().item(), LengthUnit.cm)
            # patient.get_blood_volume_baseline().set_value(s0[i, s0_cols.index("blood_volume")].detach().item(), VolumeUnit.mL)
            reset_extreme_readings(patient)
        return pool

    def administer_vasopressors(self, pool, rate):
        for e in pool.get_engines().values():
            if e.is_active:
                vp_rate = rate[e.get_id() - 1].detach().item()
                weight = e.engine_initialization.patient_configuration.get_patient().get_weight().get_value()
                if weight is None:
                    weight = 80.0
                infusion = SESubstanceInfusion()
                infusion.set_comment("Patient receives an infusion of Epinephrine")
                infusion.set_substance("Epinephrine")
                infusion.get_rate().set_value(vp_rate*weight, VolumePerTimeUnit.mL_Per_min)
                infusion.get_concentration().set_value(0.001, MassPerVolumeUnit.from_string("g/L"))
                e.actions.append(infusion)

    def administer_iv(self, pool, dose):
        for e in pool.get_engines().values():
            if e.is_active:
                dose_value = dose[e.get_id() - 1].detach().item()
                substance_compound = SESubstanceCompoundInfusion()
                substance_compound.set_comment("Patient receives infusion of Saline")
                substance_compound.set_compound("Saline")
                substance_compound.get_rate().set_value(dose_value/60, VolumePerTimeUnit.mL_Per_min)
                substance_compound.get_bag_volume().set_value(dose_value, VolumeUnit.mL)
                e.actions.append(substance_compound)

    def ventilation(self, pool, mechvent):
        for e in pool.get_engines().values():
            if e.is_active and mechvent[e.get_id() - 1].detach().item() == 1:
                ventilation = SEMechanicalVentilation()
                ventilation.set_comment("Patient is placed on a mechanical ventilator")
                ventilation.get_flow().set_value(50, VolumePerTimeUnit.mL_Per_s)
                ventilation.get_pressure().set_value(.2, PressureUnit.psi)
                ventilation.set_state(eSwitch.On)
                e.actions.append(ventilation)

    def pull_data(self, pool, batch_size):
        data = []
        active = torch.ones(batch_size)
        for e in pool.get_engines().values():
            if e.is_active:
                data.append(torch.tensor(list(e.data_requested.values.values())))
            else:
                data.append(torch.ones(self.st_vec_dim+1)*-1)
                active[e.get_id() - 1] = 0
        return torch.stack(data)[:,1:], active.float()

    def model(
        self,
        mini_batch,
        static_data,
        actions_obs,
        mini_batch_mask,
        mini_batch_seq_lengths,
        mini_batch_reversed,
    ):
        # T_max = mini_batch.size(1)
        T_max = 2
        logging.info("Starting pulse engine")
        pyro.module("pulse", self)
        with pyro.plate("s_minibatch", len(mini_batch)):
            st_gender = pyro.sample(
                f"s0_gender",
                dist.Categorical(logits=self.s0_gender),
                obs=static_data[:, static_cols.index("gender")],
            ).to(torch.long)
            st_age = pyro.sample(
                f"s0_age",
                dist.LogNormal(loc=self.s0_age[st_gender, 0], scale=torch.square(self.s0_age[st_gender, 1]) + 0.01),
                obs=static_data[:, static_cols.index("age")],
            )
            weight_loc, weight_scale = self.s0_weight(torch.column_stack((st_gender, st_age)))
            st_weight = pyro.sample(
                f"s0_weight",
                dist.LogNormal(loc=weight_loc.squeeze(), scale=torch.square(weight_scale.squeeze()) + 0.01),
                obs=static_data[:, static_cols.index("Weight_kg")],
            )
            s0_loc, s0_scale = self.s0_Network(torch.column_stack((st_gender, st_age, st_weight)))
            s0 = pyro.sample(f"s0", dist.LogNormal(loc=s0_loc.squeeze(), scale=torch.square(s0_scale).squeeze()+0.01).to_event(1))
            pool = self.create_pool(st_gender, st_age, st_weight, s0)
            if not pool.initialize_engines():
                logging.info("Unable to load/stabilize any engine")
                return
            st, active = self.emission(pool, 0, mini_batch, mini_batch_mask)
            for t in range(T_max - 1):
                vp_rate = torch.square(self.vaso(torch.column_stack((st, static_data))).squeeze())
                vp = pyro.sample(f"vp{t}",
                    dist.Normal(vp_rate, 0.001).mask(mini_batch_mask[:,t+1]*active),
                    obs=actions_obs[:,t,action_cols.index("median_dose_vaso")]
                )
                self.administer_vasopressors(pool, vp)
                iv_dose = torch.square(self.iv(torch.column_stack((st, static_data))).squeeze())
                iv = pyro.sample(f"iv{t}",
                    dist.Normal(iv_dose, 0.001).mask(mini_batch_mask[:,t+1]*active),
                    obs=actions_obs[:,t,action_cols.index("input_1hourly")]
                )
                self.administer_iv(pool, iv)
                mechvent_logits = self.mechvent(torch.column_stack((st, static_data))).squeeze()
                mechvent = pyro.sample(f"vent{t}",
                    dist.Categorical(logits=mechvent_logits).mask(mini_batch_mask[:,t+1]*active),
                    obs=actions_obs[:,t,action_cols.index("mechvent")]
                )
                self.ventilation(pool, mechvent)
                pool.process_actions()
                pool.advance_time_s(30)
                st, active = self.emission(pool, t+1, mini_batch, mini_batch_mask)

    def guide(
        self,
        mini_batch,
        static_data,
        actions_obs,
        mini_batch_mask,
        mini_batch_seq_lengths,
        mini_batch_reversed,
    ):
        # T_max = mini_batch.size(1)
        T_max = 1
        # if on gpu we need the fully broadcast view of the rnn initial state
        # to be in contiguous gpu memory
        h_0_contig = self.h_0.expand(
            1, mini_batch.size(0), self.rnn.hidden_size
        ).contiguous()
        # push the observed x's through the rnn;
        # rnn_output contains the hidden state at each time step
        rnn_output, _ = self.rnn(mini_batch_reversed, h_0_contig)
        # reverse the time-ordering in the hidden state and un-pack it
        rnn_output = poly.pad_and_reverse(rnn_output, mini_batch_seq_lengths)
        # st_prev = torch.column_stack(
        #     (self.s_q_0.expand(mini_batch.size(0), self.s_q_0.size(0)), static_data)
        # )
        pyro.module("pulse", self)
        with pyro.plate("s_minibatch", len(mini_batch)):
            # for t in range(T_max):
            st_loc, st_scale = self.combiner(static_data, rnn_output[:, 0, :])
            st_hr = pyro.sample(
                f"s0",
                dist.LogNormal(
                    st_loc.squeeze(), torch.exp(st_scale).squeeze() + 0.01
                ).to_event(1),
            )
            # st_prev = torch.column_stack((st_hr.unsqueeze(1), static_data))