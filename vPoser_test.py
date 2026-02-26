import torch
import numpy as np
from os import path as osp
from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.tools.model_loader import load_model
from human_body_prior.models.vposer_model import VPoser
from human_body_prior.tools.omni_tools import copy2cpu as c2c
import trimesh

'''
this file effectively adapts the already given .ipynb code into a .py style
'''

'''
this code seems to be reading the sample .npz file under the VPoserModelFiles folder. The
main dataset to be trained/tested on is AMASS_CMUsubset(?)
'''

def path_setup(vposer_dir, amass_dir, sub_pose):
    expr_dir = osp.join(vposer_dir, 'vposer_v2_05/') 
    
    sample_amass_fname = osp.join(amass_dir, sub_pose)  
    
    print(f"Loading Model from: {expr_dir}")
    print(f"Loading Data from: {sample_amass_fname}")
    
    return expr_dir, sample_amass_fname

def body_model_loading(bm_fname, device):
    #Loading SMPLx Body Model
    bm = BodyModel(bm_fname=bm_fname).to(device)
    return bm


def load_vposer(expr_dir, device):
    #Loading VPoser VAE Body Pose Prior
    vp, ps = load_model(expr_dir, model_code=VPoser,
                              remove_words_in_model_weights='vp_model.',
                              disable_grad=True,
                              comp_device=device)
    vp = vp.to(device)
    return vp, ps


def amass_poses_load(sample_amass_fname, device):
    # Prepare the body poses from amass sample file
    #  indexing [3:66] removes global rotation, hands/fingers, and anything else other than 21 major body joints
    amass_body_pose = np.load(sample_amass_fname)['poses'][:, 3:66]
    amass_body_pose = torch.from_numpy(amass_body_pose).type(torch.float).to(device)
    print('amass_body_pose.shape', amass_body_pose.shape)
    return amass_body_pose


def pose_encode(vp, amass_body_pose):
    # run the encoder on all frames
    amass_body_poZ = vp.encode(amass_body_pose).mean
    print('amass_body_poZ.shape', amass_body_poZ.shape)
    return amass_body_poZ


def pose_decode(vp, amass_body_poZ):
    # run the decoder on all frames
    amass_body_pose_rec = vp.decode(amass_body_poZ)['pose_body'].contiguous().view(-1, 63)
    print('amass_body_pose_rec.shape', amass_body_pose_rec.shape)
    return amass_body_pose_rec


def get_mesh_for_pose(bm, amass_body_pose, amass_body_pose_rec):
    #get vertices and faces of a polygonal mesh model for each body pose
    originalPoses = {'pose_body':amass_body_pose}
    recoveredPoses = {'pose_body':amass_body_pose_rec}
    
    bmodelorig = bm(**originalPoses)
    bmodelreco = bm(**recoveredPoses)
    vorig = c2c(bmodelorig.v)
    vreco = c2c(bmodelreco.v)
    faces = c2c(bm.f)
    
    T, num_verts = vorig.shape[:-1]
    
    return vorig, vreco, faces, bmodelorig, bmodelreco


def visualize_single_frame(vorig, vreco, faces, fIdx):
    #visualize one frame's body pose before (grey) and after (purple) encode-decode
    verts = vorig[fIdx]
    mesh1 = trimesh.base.Trimesh(verts, faces)
    mesh1.visual.vertex_colors = [254, 254, 254]
    verts = vreco[fIdx]
    mesh2 = trimesh.base.Trimesh(verts, faces)
    mesh2.visual.vertex_colors = [254, 66, 200]
    mesh2.apply_translation([1, 0, 0])  #use [0,0,0] to overlay them on each other
    meshes = [mesh1, mesh2]
    trimesh.Scene(meshes).show()


def temporal_seq_visualize(vorig, vreco, faces):
    #visualize a temporal subsequence of poses spatially (use mouse to rotate view)
    #note that encoding followed by decoding is not a lossless process,
    #it can introduce a certain amount of error all by itself
    meshes = []
    for fIdx in range(0,200,10):
        verts = vorig[fIdx]
        mesh1 = trimesh.base.Trimesh(verts, faces)
        mesh1.visual.vertex_colors = [254, 254, 254]
        mesh1.apply_translation([0, 0, fIdx*.07])
        meshes.append(mesh1)
        verts = vreco[fIdx]
        mesh1 = trimesh.base.Trimesh(verts, faces)
        mesh1.visual.vertex_colors = [254, 150, 200]
        mesh1.apply_translation([0, 0, fIdx*.07])
        meshes.append(mesh1)
    
    trimesh.Scene(meshes).show()


def visualize_joints(vorig, vreco, faces, bmodelorig, bmodelreco, fIdx):
    # extract and visualize 23 body joints before and after encode-decode process
    # for a pose where error between original pose and decoded pose is rather large.
    # why 23 instead of 21 mentioned earlier?  There are two extra joints somewhere
    # that are not among the 21 rotatable body joints used by VPoser.
    verts = vorig[fIdx]
    mesh1 = trimesh.base.Trimesh(verts, faces)
    mesh1.visual.vertex_colors = [254, 254, 254]
    verts = vreco[fIdx]
    mesh2 = trimesh.base.Trimesh(verts, faces)
    mesh2.visual.vertex_colors = [254, 66, 200]
    mesh2.apply_translation([0, 0, 0])  #use [0,0,0] to overlay them on each other
    meshes = [mesh1, mesh2]
    
    #get the 23 major 3D body joints
    joints = c2c(bmodelorig.Jtr[fIdx])
    origjoints = joints[0:23, :]   #ignore finger joints
    joints = c2c(bmodelreco.Jtr[fIdx])
    recojoints = joints[0:23, :]   #ignore finger joints
    
    print(origjoints.shape, recojoints.shape)
    for i in range(origjoints.shape[0]):
        sphere = trimesh.primitives.Sphere(radius=.02, center=origjoints[i,:])
        sphere.apply_translation([1, 0, 0])
        sphere.visual.vertex_colors = [254, 254, 254]
        meshes.append(sphere)
        sphere = trimesh.primitives.Sphere(radius=.02, center=recojoints[i,:])
        sphere.apply_translation([1, 0, 0])
        sphere.visual.vertex_colors = [254, 150, 200]
        meshes.append(sphere)
    
    trimesh.Scene(meshes).show()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    support_dir = 'VPoserModelFiles'
    expr_dir, bm_fname, sample_amass_fname = path_setup(support_dir)
    
    bm = body_model_loading(bm_fname, device)
    
    vp, ps = load_vposer(expr_dir, device)
    
    amass_body_pose = amass_poses_load(sample_amass_fname, device)
    
    amass_body_poZ = pose_encode(vp, amass_body_pose)
    
    amass_body_pose_rec = pose_decode(vp, amass_body_poZ)
    
    vorig, vreco, faces, bmodelorig, bmodelreco = get_mesh_for_pose(bm, amass_body_pose, amass_body_pose_rec)
    
    visualize_single_frame(vorig, vreco, faces, 140)
    
    temporal_seq_visualize(vorig, vreco, faces)
    
    visualize_joints(vorig, vreco, faces, bmodelorig, bmodelreco, 130)


if __name__ == "__main__":
    main()
