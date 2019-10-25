classdef MultiPeopleTriPose
    % Triangulates all people in a scene
    
    properties
        tri_poses
        nbr_people
    end
    
    methods
        
        function obj = MultiPeopleTriPose(nbr_people)
            obj.nbr_people = nbr_people;
            obj.tri_poses = cell(1, obj.nbr_people);
            for i = 1 : obj.nbr_people
                obj.tri_poses{i} = TriangulatedPose();
            end
        end
        
        function reset(obj)
            for i = 1 : obj.nbr_people
                obj.tri_poses{i}.reset();
            end
        end
        
        function remove(obj)
            for i = 1 : obj.nbr_people
                obj.tri_poses{i}.remove();
            end
        end
    end
end
