import { Box } from "@mantine/core";
import { Chessboard } from "react-chessboard";

export const ChessGame = () => {
  return (
    <Box w="48rem" h="48rem">
      <Chessboard
        position={"start"}
        onPieceDragEnd={(piece, square) => {
          console.log(piece, square);
        }}
      />
    </Box>
  );
};
