import { useState, useEffect } from 'react'
import { styled } from '@mui/material/styles';
import { Button, IconButton } from '@mui/material';
import Stack from '@mui/material/Stack';
import SearchIcon from '@mui/icons-material/Search';
import CameraAltIcon from '@mui/icons-material/CameraAlt';
import eraser from '../icons/eraser.png'
import './style.css'
import Info from './Info'
import { useNavigate } from "react-router-dom";
import ArrowForwardIcon from '@mui/icons-material/ArrowForward';


const VisuallyHiddenInput = styled('input')({
    clip: 'rect(0 0 0 0)',
    clipPath: 'inset(50%)',
    height: 1,
    overflow: 'hidden',
    position: 'absolute',
    bottom: 0,
    left: 0,
    whiteSpace: 'nowrap',
    width: 1,
  });


export default function Home() {
    const [popup, setPopup] = useState(false)
    const [fileName, setFileName] = useState(null);
    const navigate = useNavigate();
    var [fileCount, setFileCount] = useState(0);

    const handleFileUpload = (event) => {
        const file = event.target.files[0];
        setFileName(file.name);
        setFileCount(1)
        // postFileName(file.name);
    };

    // const postFileName = async (fileName) => {
    //     const reqOptions = {
    //         method: 'POST',
    //         headers: { "Content-Type": "application/json" },
    //         body: JSON.stringify({"fileName": fileName})
    //     }

    //     const res = await fetch("http://localhost:5000/fileName", reqOptions);
    //     const resJson = await res.json();
    //     fileCount = fileCount + 1;
    //     setFileCount(fileCount)
    //     console.log(resJson)
    // }

    const InfoPopupHandler = () => {
        setPopup(true)
    }
    
    return(
        <div className = 'wrapper'>
            <Stack direction="row">
                <IconButton aria-label="delete" 
                            sx = {{marginRight: 30, marginTop: -3, color: 'black', justifyContent: 'flex-end'}}
                            onClick = {() => InfoPopupHandler()}>
                <img  src={eraser} style = {{marginLeft: 30, marginTop: 30}}/>
                </IconButton>
            </Stack>
            <div style = {{marginTop: "200px"}}>
                <h1 style = {{fontFamily:"Roboto", fontSize: "3.6rem", fontWeight: "bold"}}>Did We Train You?</h1>

                {(fileCount === 0)
                ?
                <Button component="label" variant="contained" 
                        startIcon={<SearchIcon style={{marginLeft: 20, marginRight: 5, fontSize: 50}}/>} 
                        endIcon={<CameraAltIcon style={{marginLeft: 240, fontSize: 50}}/>}
                        sx = {{width: 600, height: 80, borderRadius: "50px", justifyContent: "flex-start", 
                            fontSize: "1.5rem", fontFamily: "Roboto", textTransform: 'none', fontWeight: "200px", marginTop: "-15px"}}
                        >
                    Upload Image
                    
                    <input
                        id="upload-image"
                        hidden
                        accept="image/*"
                        type="file"
                        onChange={handleFileUpload}
                    />
                </Button>
                :
                <Button component="label" variant="contained" color="secondary"
                        startIcon={<SearchIcon style={{marginLeft: 20, marginRight: 5, fontSize: 50}}/>} 
                        endIcon={<ArrowForwardIcon style={{marginLeft: 240, fontSize: 50}}/>}
                        sx = {{width: 600, height: 80, borderRadius: "50px", justifyContent: "flex-start", 
                            fontSize: "1.5rem", fontFamily: "Roboto", textTransform: 'none', fontWeight: "200px", marginTop: "-15px"}}
                        onClick = {() => navigate("/ModelCheck/"+fileName)}
                        >
                    Check Model
                </Button>
                }

                <p style ={{fontSize: "1.5rem", fontFamily: "Roboto"}}>Check if your image has been trained by our AI model</p>
            </div>
            {(popup) && <Info setPopup = {setPopup}/>}
        </div>


    )
}
