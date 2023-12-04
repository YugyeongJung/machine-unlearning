import './style.css'
import { Fragment, useState, useEffect } from 'react'
import CloseOutlinedIcon from '@mui/icons-material/CloseOutlined';
import { Button, IconButton, Stack } from '@mui/material';
import blackFrame from '../icons/blackFrame.png'
import grayFrame from '../icons/grayFrame.png'
import eraser from '../icons/eraser.png'
import data from '../icons/data.png'
import check from '../icons/check.png'

export default function Help(props){

    const [lan, setLan] = useState('Eng')
    const translateHandlerKor = () => {
        setLan('Kor')
    }
    const translateHandlerEng = () => {
        setLan('Eng')
    }

    return(
        <Fragment>
        <div className='popup-shadow'>
            <div className='popup-help-attack'>
                <h1 style  = {{fontSize: 30, fontFamily: "Roboto", color: "#247AFC"}}>Model Inversion Attack (MI ATTACK) 
                    <IconButton sx = {{marginLeft: 90, marginTop: -2}} onClick = {() => props.setPopup(false)}>
                        <CloseOutlinedIcon/>
                    </IconButton>
                    <hr />
                </h1>
                <div style = {{fontSize: 25, fontFamile: "Roboto", color: "black", fontWeight: 10, textAlign: "left", marginLeft: 5}}>
                
                {(lan == 'Eng')
                ?
                <p>
                Model Inversion Attack (MI Attack) is a method used to extract training data from AI models. We apply MI attack before and after unlearning your data. Before unlearning, our resulting image shows the representation of your data and similar (same class) training data. After unlearning, our resulting image is the representation of similar (same class) training data without your data. Therefore, the difference between these two images is your data deleted from the AI model.
                </p>
                :
                <p>
                Model Inversion Attack (MI ATTACK)은 AI 모델에서 학습 데이터를 추출하는 데 사용되는 방법입니다. 언러닝 전과 언러닝 후, 총 2번 MI Attack을 적용합니다. 언러닝 전에 MI Attack을 통해 추출된 이미지는 사용자 데이터 & 유사한(동일한 클래스) 데이터의 흔적을 보여주고 있습니다. 언러닝 후 MI Attack으로 얻은 이미지는 사용자 데이터를 제외한 유사한(동일한 클래스) 데이터입니다. 그리고 이 두 이미지의 차이가 바로 AI 모델에서 지워진 사용자의 데이터 흔적입니다. </p>
                }

                </div>
                <hr />
                <Stack direction = "row" justifyContent = "center" sx = {{marginTop: 2}}>
                    <Button variant = "contained" sx = {{width: 100, height: 40}} onClick = {() => setLan('Kor')}>Kor</Button>
                    <Button variant = "outlined" sx = {{marginLeft: 1, width: 100, height: 40}} onClick = {() => setLan('Eng')}>Eng</Button>
                </Stack>

            </div>         
        </div>
        </Fragment>
    )
}
